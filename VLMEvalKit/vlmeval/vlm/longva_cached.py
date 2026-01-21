"""
LongVA 模型 - 支持 Vision 特征缓存
通过缓存 vision_tower 输出，避免重复计算 vision encoder
mm_projector 和 LLM 正常运行，确保流程完全一致
"""

import torch
from PIL import Image
import numpy as np
import warnings
import sys
import os
from pathlib import Path
import hashlib

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

from .base import BaseModel


class LongVA_Cached(BaseModel):
    """
    LongVA model wrapper with vision feature caching support.

    缓存 vision_tower 的输出，mm_projector 和 LLM 正常运行。
    确保除 vision_tower 外的流程与原始 LongVA 完全一致。

    使用方法:
        1. 先运行 preprocess_video_mme.py 预处理数据集
        2. 在 config.py 中配置:
           "LongVA-7B-Cached": partial(
               LongVA_Cached,
               model_path="lmms-lab/LongVA-7B-DPO",
               cache_dir="/disk3/minami/LMUData/vision_cache"
           )
    """

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(self, model_path="lmms-lab/LongVA-7B",
                 cache_dir=None,
                 enable_cache=True,
                 **kwargs):
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

        # 缓存配置
        self.enable_cache = enable_cache and cache_dir is not None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_hits = 0
        self.cache_misses = 0

        # 加载模型
        print(f"加载模型: {model_path}")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="auto"
        )
        self.model.eval()

        # 生成模型签名
        self.model_signature = self._generate_model_signature()

        # 默认生成配置
        self.gen_kwargs = {
            "do_sample": False,
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": 1024,
        }

        # 视频配置
        self.nframe = kwargs.get('nframe', 128)  # 默认 128 帧，与预处理一致
        self.fps = kwargs.get('fps', -1)

        torch.cuda.empty_cache()

        if self.enable_cache:
            print(f"✓ Vision 缓存已启用: {self.cache_dir}")
            print(f"✓ 模型签名: {self.model_signature}")
            print(f"✓ 缓存模式: float16 无压缩")

    def _generate_model_signature(self) -> str:
        """生成模型签名（基于 vision tower 配置）"""
        config = self.model.config
        sig_str = f"{config.mm_vision_tower}:layer{config.mm_vision_select_layer}"
        return hashlib.md5(sig_str.encode()).hexdigest()[:8]

    def _get_cache_key(self, video_id: str, dataset: str, nframe: int) -> str:
        """生成缓存键"""
        key_str = f"{dataset}:{video_id}:nframe{nframe}:model{self.model_signature}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, dataset: str) -> Path:
        """获取缓存文件路径"""
        dataset_dir = self.cache_dir / dataset
        return dataset_dir / f"{cache_key}.pt"

    def _load_cached_vision_features(self, video_path: str, dataset: str,
                                     nframe: int) -> torch.Tensor:
        """加载缓存的 vision_tower 输出"""
        if not self.enable_cache:
            return None

        video_id = Path(video_path).stem
        cache_key = self._get_cache_key(video_id, dataset, nframe)
        cache_path = self._get_cache_path(cache_key, dataset)

        if not cache_path.exists():
            return None

        try:
            data = torch.load(cache_path, map_location='cpu')
            vision_features = data['vision_features']

            # 移动到设备并保持 float16
            vision_features = vision_features.to(self.model.device, dtype=torch.float16)

            return vision_features

        except Exception as e:
            print(f"警告: 加载缓存失败 {cache_path}: {e}")
            return None

    def _load_video_frames(self, video_path: str, max_frames=None):
        """加载视频帧（与 longva_custom.py 完全一致）"""
        from decord import VideoReader, cpu

        if max_frames is None:
            max_frames = self.nframe

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)

        # 均匀采样（与 longva_custom.py 完全一致）
        if total_frame_num <= max_frames:
            frame_idx = list(range(total_frame_num))
        else:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

        frames = vr.get_batch(frame_idx).asnumpy()
        return frames

    def use_custom_prompt(self, dataset):
        """Check if custom prompt should be used for a dataset."""
        return False

    def build_prompt(self, line, dataset=None):
        """Build prompt from dataset line."""
        import pandas as pd
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

    def generate_inner(self, message, dataset=None):
        """Generate response with vision feature caching support."""
        # 提取视频路径和文本
        video_path = None
        video_frames = None
        prompt_text = ""
        has_video = False

        for item in message:
            if item["type"] == "video":
                video_path = item["value"]
                has_video = True
            elif item["type"] == "image":
                # 图像模式，使用原始实现
                return self._generate_image_mode(message)
            elif item["type"] == "text":
                if prompt_text:
                    prompt_text += "\n" + item["value"]
                else:
                    prompt_text = item["value"]

        if not has_video or video_path is None:
            # 纯文本模式
            return self._generate_text_mode(prompt_text)

        # 视频模式：尝试使用缓存
        cached_vision_features = self._load_cached_vision_features(
            video_path, dataset, self.nframe
        )

        if cached_vision_features is not None:
            # 缓存命中：使用 hook 方式
            self.cache_hits += 1
            return self._generate_with_cached_features(
                video_path, cached_vision_features, prompt_text
            )
        else:
            # 缓存未命中：正常推理
            self.cache_misses += 1
            return self._generate_without_cache(video_path, prompt_text)

    def _generate_with_cached_features(self, video_path: str,
                                       cached_vision_features: torch.Tensor,
                                       prompt_text: str) -> str:
        """使用缓存的 vision features 生成（通过 hook）"""

        # 获取 vision_tower
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
            vision_tower = self.model.model.vision_tower
        elif hasattr(self.model, 'vision_tower'):
            vision_tower = self.model.vision_tower
        else:
            # Fallback：无法 hook，使用正常推理
            print("警告: 无法访问 vision_tower，使用正常推理")
            return self._generate_without_cache(video_path, prompt_text)

        # 保存原始 forward 方法
        original_forward = vision_tower.forward

        # 定义 hook：返回缓存的特征
        def cached_forward(x):
            # 注意：需要添加 batch 维度
            return cached_vision_features.unsqueeze(0)

        try:
            # 替换 forward 方法
            vision_tower.forward = cached_forward

            # 🚀 优化：不加载真实视频，创建 dummy tensor
            # 只需要正确的 shape，内容不重要（因为 vision_tower 会返回缓存）
            # Shape: [nframe, 3, height, width]
            dummy_video_tensor = torch.zeros(
                (self.nframe, 3, 336, 336),
                dtype=torch.float16,
                device=self.model.device
            )

            # 构建 prompt（与 longva_custom.py:149 一致）
            prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            input_ids = self.tokenizer_image_token(
                prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.model.device)

            # 生成（vision_tower 会返回缓存的特征，mm_projector 和 LLM 正常运行）
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=[dummy_video_tensor],  # 传入 dummy tensor（不会真正使用）
                    modalities=["video"],
                    **self.gen_kwargs
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs

        finally:
            # 恢复原始 forward 方法
            vision_tower.forward = original_forward

    def _generate_without_cache(self, video_path: str, prompt_text: str) -> str:
        """正常推理（无缓存，与 longva_custom.py 完全一致）"""
        # 加载视频帧（与 longva_custom.py:125 一致）
        frames = self._load_video_frames(video_path)

        # 预处理（与 longva_custom.py:145 一致）
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.to(self.model.device, dtype=torch.float16)

        # 构建 prompt（与 longva_custom.py:149 一致）
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        # 生成（与 longva_custom.py:156-162 一致）
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                **self.gen_kwargs
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

    def _generate_text_mode(self, prompt_text: str) -> str:
        """纯文本模式（与 longva_custom.py:134-140 一致）"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, **self.gen_kwargs)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

    def _generate_image_mode(self, message) -> str:
        """图像模式（与 longva_custom.py:167-196 一致）"""
        images = []
        prompt_text = ""

        for item in message:
            if item["type"] == "image":
                img = Image.open(item["value"]).convert("RGB")
                images.append(img)
            elif item["type"] == "text":
                if prompt_text:
                    prompt_text += "\n" + item["value"]
                else:
                    prompt_text = item["value"]

        # 处理图像（与 longva_custom.py:169-172 一致）
        images_tensor = self.process_images(images, self.image_processor, self.model.config)
        if isinstance(images_tensor, list):
            images_tensor = torch.stack(images_tensor, dim=0)
        images_tensor = images_tensor.to(self.model.device, dtype=torch.float16)

        # 构建 prompt（与 longva_custom.py:175-176 一致）
        image_tokens = "<image>" * len(images)
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_tokens}\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        # 获取图像尺寸（与 longva_custom.py:183 一致）
        image_sizes = [img.size for img in images]

        # 生成（与 longva_custom.py:186-193 一致）
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

    def __del__(self):
        """析构时输出缓存统计"""
        if self.enable_cache and (self.cache_hits + self.cache_misses) > 0:
            total = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total * 100
            print(f"\n{'='*60}")
            print("Vision Cache Statistics")
            print(f"{'='*60}")
            print(f"Cache hits: {self.cache_hits}")
            print(f"Cache misses: {self.cache_misses}")
            print(f"Hit rate: {hit_rate:.1f}%")
            print(f"{'='*60}")
