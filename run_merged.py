from safetensors.torch import load_file
import torch, gc
from PIL import Image
from longva.model.builder import load_pretrained_model
from longva.mm_utils    import tokenizer_image_token, process_images
from longva.constants   import IMAGE_TOKEN_INDEX

# 1. 载入 LongVA
tokenizer, model, image_processor, _ = load_pretrained_model(
        "lmms-lab/LongVA-7B-DPO", None, "llava_qwen", device_map="cuda:0"
)
model.to(torch.bfloat16)

# 2. 覆写 delta（只覆写 transformer）
delta = load_file("qwen2_longva_delta/delta.safetensors")
model.load_state_dict(delta, strict=False)      # missing 很多是正常的
del delta; gc.collect()

# 3. 构造输入
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<image>\n"
    "Find the frame of a couple in a wedding. Inside the frame, "
    "there is a balloon on the bridegroom's head. What is the color "
    "of that balloon? Answer using a single word or phrase."
    "<|im_end|>\n<|im_start|>assistant\n"
)

input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(model.device)

img = Image.open("image/needle.JPG").convert("RGB")
img_tensor = process_images([img], image_processor, model.config).to(model.device, dtype=torch.bfloat16)

# 4. 生成
gen_kwargs = dict(do_sample=True, temperature=0.5,
                  num_beams=1, max_new_tokens=64)

with torch.inference_mode():
    out = model.generate(
        input_ids,
        images       = img_tensor,
        image_sizes  = [img.size],
        modalities   = ["image"],
        **gen_kwargs
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
