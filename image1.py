import torch
from PIL import Image, ImageDraw
import numpy as np
from transformers import CLIPImageProcessor

def compute_and_visualize(image_path, bbox, image_processor, grid_size=24):
    # 加载原图
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    print(f"[INFO] 原始图片大小: {orig_w} x {orig_h}")

    # 处理图片
    processed = image_processor(image, return_tensors='pt')['pixel_values']
    _, _, processed_h, processed_w = processed.shape
    print(f"[INFO] 处理后图片大小: {processed_w} x {processed_h}")

    # 计算缩放比例
    x_scale, y_scale = processed_w / orig_w, processed_h / orig_h
    xmin, ymin, xmax, ymax = bbox
    scaled_xmin, scaled_ymin = xmin * x_scale, ymin * y_scale
    scaled_xmax, scaled_ymax = xmax * x_scale, ymax * y_scale
    print(f"[DEBUG] Scaled BBox: ({scaled_xmin:.2f}, {scaled_ymin:.2f}, {scaled_xmax:.2f}, {scaled_ymax:.2f})")

    # Grid大小
    grid_w, grid_h = processed_w / grid_size, processed_h / grid_size

    # 计算覆盖的 grid 范围
    grid_x_min = int(np.floor(scaled_xmin / grid_w))
    grid_x_max = int(np.ceil(scaled_xmax / grid_w))
    grid_y_min = int(np.floor(scaled_ymin / grid_h))
    grid_y_max = int(np.ceil(scaled_ymax / grid_h))
    print(f"[DEBUG] Grid X: {grid_x_min}-{grid_x_max}, Grid Y: {grid_y_min}-{grid_y_max}")

    # 记录覆盖到的 visual tokens
    selected_tokens = []
    for gy in range(grid_y_min, grid_y_max):
        for gx in range(grid_x_min, grid_x_max):
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                token_idx = gy * grid_size + gx
                selected_tokens.append(token_idx)
    print(f"[DEBUG] 选中的 Visual Tokens: {selected_tokens}")

    # ✅ 可视化1：画出选中的 grid 和缩放后的bbox
    vis = Image.new("RGB", (processed_w, processed_h), "black")
    draw = ImageDraw.Draw(vis)
    for token in selected_tokens:
        gx, gy = token % grid_size, token // grid_size
        draw.rectangle(
            [gx * grid_w, gy * grid_h, (gx + 1) * grid_w, (gy + 1) * grid_h],
            fill="white"
        )
    draw.rectangle([scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax], outline="red", width=3)
    vis.save("visual_token_selection.png")
    print("[INFO] 可视化1已保存为 visual_token_selection.png")

    # ✅ 反推验证：从 token 反推 grid 和像素区域
    vis_reverse = Image.new("RGB", (processed_w, processed_h), "black")
    draw_rev = ImageDraw.Draw(vis_reverse)
    for token in selected_tokens:
        gy, gx = divmod(token, grid_size)
        pixel_x_min = gx * grid_w
        pixel_y_min = gy * grid_h
        pixel_x_max = (gx + 1) * grid_w
        pixel_y_max = (gy + 1) * grid_h
        draw_rev.rectangle([pixel_x_min, pixel_y_min, pixel_x_max, pixel_y_max], fill="white")

    draw_rev.rectangle([scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax], outline="red", width=3)
    vis_reverse.save("reverse_token_mapping.png")
    print("[INFO] 可视化2（反推）已保存为 reverse_token_mapping.png")

    return selected_tokens

# === 直接跑起来 ===
image_path = "target.JPG"
bbox = (1000, 2270, 2357, 2802)  # 原始图片坐标
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

compute_and_visualize(image_path, bbox, image_processor)
