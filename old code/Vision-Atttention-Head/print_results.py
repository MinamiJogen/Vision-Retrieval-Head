import numpy as np

def inspect_npz(npz_file):
    # 加载 NPZ 文件
    data = np.load(npz_file, allow_pickle=True)
    print(f"Loaded NPZ file with {len(data.keys())} entries.\n")
    
    # 遍历 NPZ 文件中的所有 key
    for sample_id in data.keys():
        sample_data = data[sample_id].item()
        print(f"Sample ID: {sample_id}")
        
        # 打印 query 和 answer 以及生成文本
        question = sample_data.get("question", "N/A")
        answer = sample_data.get("answer", "N/A")
        generated_text = sample_data.get("generated_text", "N/A")
        print(f"  Question: {question}")
        print(f"  Answer: {answer}")
        print(f"  Generated Text: {generated_text}")
        
        # 打印 split_index 和 input token 的长度
        split_index = sample_data.get("split_index", None)
        input_token_length = sample_data.get("input_token_length", None)
        print(f"  Split Index: {split_index}")
        print(f"  Input Token Length: {input_token_length}")
        
        # 打印 attention 层的基本信息（不详细打印每层内容）
        attn_layers = sample_data.get("attention", {})
        if not isinstance(attn_layers, dict):
            print("  ⚠️ Invalid attention format. Skipping...\n")
            continue
        
        print(f"  Number of Attention Layers: {len(attn_layers)}")
        for layer, attn in attn_layers.items():
            # 如果是 tensor 则转换为 numpy 数组
            if hasattr(attn, 'numpy'):
                attn = attn.numpy()
            print(f"    Layer: {layer}, Shape: {attn.shape}")
        
        print("-" * 80)

if __name__ == "__main__":
    npz_file = "LongVA_attention_analysis/MTVQA.npz"
    inspect_npz(npz_file)
