import torch
from PIL import Image
import numpy as np
import warnings
import sys
import os

# Suppress PyTorch meta parameter warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*for.*copying from a non-meta parameter.*")

from .base import BaseModel


class LongVA(BaseModel):
    """
    LongVA model wrapper for VLMEvalKit.
    Supports LongVA-7B, Temporal-v1, and Temporal-v2 models.
    Supports both image and video inputs.
    """

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True  # Enable video support

    def __init__(self, model_path="lmms-lab/LongVA-7B-DPO", **kwargs):
        try:
            from longva.model.builder import load_pretrained_model
            from longva.mm_utils import tokenizer_image_token, process_images
            from longva.constants import IMAGE_TOKEN_INDEX
        except ImportError:
            raise ImportError(
                "LongVA is not installed. Please install it from the LongVA repository."
            )

        self.model_path = model_path
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX

        # Load model
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="auto"
        )
        self.model.eval()

        # Default generation config
        self.gen_kwargs = {
            "do_sample": False,
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": 1024,
        }

        # Video config
        self.nframe = kwargs.get('nframe', 64)
        self.fps = kwargs.get('fps', -1)

        torch.cuda.empty_cache()

    def use_custom_prompt(self, dataset):
        """Check if custom prompt should be used for a dataset."""
        return False

    def build_prompt(self, line, dataset=None):
        """Build prompt from dataset line."""
        from ..smp import listinstr

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        if "options" in line and not pd.isna(line["options"]):
            options = eval(line["options"]) if isinstance(line["options"], str) else line["options"]
            if isinstance(options, list):
                options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                question = f"{question}\n{options_str}"
            elif isinstance(options, dict):
                options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
                question = f"{question}\n{options_str}"

        msgs = [dict(type="image", value=p) for p in tgt_path]
        msgs.append(dict(type="text", value=question))
        return msgs

    def _load_video_frames(self, video_path, max_frames=None):
        """Load video frames from a video file."""
        from decord import VideoReader, cpu

        if max_frames is None:
            max_frames = self.nframe

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)

        # Uniformly sample frames
        if total_frame_num <= max_frames:
            frame_idx = list(range(total_frame_num))
        else:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

        frames = vr.get_batch(frame_idx).asnumpy()
        return frames

    def generate_inner(self, message, dataset=None):
        """Generate response for given message. Supports both images and videos."""
        # Extract images, videos and text from message
        images = []
        video_frames = None
        prompt_text = ""
        has_video = False

        for item in message:
            if item["type"] == "image":
                img = Image.open(item["value"]).convert("RGB")
                images.append(img)
            elif item["type"] == "video":
                # Load video frames
                video_path = item["value"]
                video_frames = self._load_video_frames(video_path)
                has_video = True
            elif item["type"] == "text":
                if prompt_text:
                    prompt_text += "\n" + item["value"]
                else:
                    prompt_text = item["value"]

        if not images and video_frames is None:
            # Text-only query (shouldn't happen for VLM benchmarks)
            prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, **self.gen_kwargs)
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs

        # Handle video input
        if has_video and video_frames is not None:
            # Process video frames
            video_tensor = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
            video_tensor = video_tensor.to(self.model.device, dtype=torch.float16)

            # Build prompt with image token (single token for video)
            prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            input_ids = self.tokenizer_image_token(
                prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.model.device)

            # Generate response for video
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=[video_tensor],
                    modalities=["video"],
                    **self.gen_kwargs
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs

        # Handle image input (original code)
        # Process images
        images_tensor = self.process_images(images, self.image_processor, self.model.config)
        if isinstance(images_tensor, list):
            images_tensor = torch.stack(images_tensor, dim=0)
        images_tensor = images_tensor.to(self.model.device, dtype=torch.float16)

        # Build prompt with image token
        image_tokens = "<image>" * len(images)
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_tokens}\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        # Get image sizes
        image_sizes = [img.size for img in images]

        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                modalities=["image"] * len(images),
                **self.gen_kwargs
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
