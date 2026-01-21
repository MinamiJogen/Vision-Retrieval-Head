import torch
from functools import partial
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from PIL import Image
import requests
from io import BytesIO

# 存储注意力分布
attention_maps = {}

def hook_attention(module, input, output, layer_idx):
    """Hook function to store attention weights from a layer."""
    if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
        attention_maps[layer_idx] = output[1].detach().cpu()

def load_image(image_file):
    """加载本地或在线图片"""
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def call_LLaVA_with_attention(query, image, tokenizer, model, image_processor):
    """
    Call LLaVA model with preloaded components to avoid redundant loading.

    Args:
    - query (str): The input query for the model.
    - image (str): The image file path or URL.
    - tokenizer: Preloaded tokenizer
    - model: Preloaded model
    - image_processor: Preloaded image processor

    Returns:
    - output_text (str): The generated response.
    - attention_maps (dict): A dictionary containing attention weights for each layer.
    """
    global attention_maps  # 确保 attention_maps 在多个调用间共享
    attention_maps = {}  # 重置注意力存储

    # 处理输入文本
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    qs = query + "\n" + image_token_se if model.config.mm_use_im_start_end else query + "\n" + DEFAULT_IMAGE_TOKEN
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 处理图像
    image = load_image(image)
    images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    
    print("Prompt for debugging:\n", prompt)

    # 找出文本和图像的分界点
    split_index = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item()

    # Hook 注册
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.register_forward_hook(partial(hook_attention, layer_idx=i))

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
            output_attentions=True,
        )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output_text, attention_maps, split_index
