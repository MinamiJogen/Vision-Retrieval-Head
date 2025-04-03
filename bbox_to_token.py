def map_bbox_to_input_token_indices(image, bbox, image_processor, model_config, vision_tower, tokenizer, prompt):
    """
    计算 bbox 对应的视觉 token 在 input_ids 中的 token 索引范围
    """
    # Step 1: 处理图片
    raw_image_tensor = image_processor(image, return_tensors='pt')['pixel_values']
    processed_h, processed_w = raw_image_tensor.shape[-2:]
    grid_size = 24  # 24x24 视觉 patch
    grid_w = processed_w / grid_size
    grid_h = processed_h / grid_size

    # Step 2: bbox 缩放到处理后尺寸
    x_scale = processed_w / image.width
    y_scale = processed_h / image.height
    scaled_xmin = bbox[0] * x_scale
    scaled_ymin = bbox[1] * y_scale
    scaled_xmax = bbox[2] * x_scale
    scaled_ymax = bbox[3] * y_scale

    # Step 3: 找到覆盖的视觉 grid
    grid_x_min = int(np.floor(scaled_xmin / grid_w))
    grid_x_max = int(np.ceil(scaled_xmax / grid_w))
    grid_y_min = int(np.floor(scaled_ymin / grid_h))
    grid_y_max = int(np.ceil(scaled_ymax / grid_h))

    visual_token_indices = []
    for gy in range(grid_y_min, grid_y_max):
        for gx in range(grid_x_min, grid_x_max):
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                visual_token_indices.append(gy * grid_size + gx)

    print(f"[BBox Mapping] Visual tokens covered by BBox: {visual_token_indices}")

    # Step 4: 编码 prompt，获取视觉 token 在 input_ids 中的起始位置
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    image_token_start = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item()
    vision_token_count = vision_tower.num_patches  # e.g., 576
    image_token_end = image_token_start + vision_token_count - 1
    print(f"[Prompt] Visual token range in input_ids: [{image_token_start}, {image_token_end}]")

    # Step 5: 把视觉 grid index 映射到 input token 中
    selected_input_token_indices = [image_token_start + idx for idx in visual_token_indices]
    print(f"[Final] Input token indices covering BBox: {selected_input_token_indices}")

    return input_ids, selected_input_token_indices
