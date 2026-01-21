import torch

def safe_tokenize(tokenizer, text):
    """
    使用 tokenizer.encode() 得到 token id，并去除 BOS token（如果存在）
    """
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token is not None and tokenized.size(1) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized

def replace_double_newline_func(token_ids):
    """
    将 token id 271 替换为两个 198
    例如：输入 tensor([[... 271, ...]]) 替换后长度会增加
    """
    # 找到所有等于 271 的位置（注意这里假设 token_ids 的 shape 为 [1, seq_len]）
    double_newline_loc = (token_ids == 271).nonzero()[:, 1]
    # 调整位置索引，确保插入不会出错
    double_newline_loc += torch.arange(len(double_newline_loc))
    if len(double_newline_loc) > 0:
        for loc in double_newline_loc:
            # 将当前位置 token 用两个 198 替换
            token_ids = torch.cat([
                token_ids[:, :loc],
                torch.tensor([[198, 198]], device=token_ids.device),
                token_ids[:, loc+1:]
            ], dim=1)
    return token_ids

def get_text_embedding(text, tokenizer, model, replace_double_newline=False, device=None):
    """
    接收字符串 text，利用 tokenizer 和模型的 embedding 层获取文本 embedding。
    
    参数：
      - text: 待转换的字符串
      - tokenizer: Hugging Face tokenizer 对象
      - model: 包含 .model.embed_tokens 方法的模型（如 Qwen2、LLaVA 等）
      - replace_double_newline: 如果为 True，则将 token id 271 替换成两个 198
      - device: 指定 device，不传时默认使用 model.device

    返回：
      - 一个 tensor，形状为 [1, seq_len, hidden_dim]，数据类型为 bfloat16
    """
    if device is None:
        device = model.device
    # 转为 token id
    token_ids = safe_tokenize(tokenizer, text)
    if replace_double_newline:
        token_ids = replace_double_newline_func(token_ids)
    token_ids = token_ids.to(device)
    with torch.inference_mode():
        embeddings = model.model.embed_tokens(token_ids)
    return embeddings.to(torch.bfloat16)

# 示例用法：
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from longva.model.builder import load_pretrained_model
    # 加载模型与 tokenizer（请根据你的实际模型路径替换 model_path）
    model_path = "lmms-lab/LongVA-7B-DPO"
    tokenizer, model, _, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto")
    
    sample_text = "What is the main object in the lower part of the second picture?"
    text_embeds = get_text_embedding(sample_text, tokenizer, model, replace_double_newline=False)
    print("Text embedding shape:", text_embeds.shape)
