import json
import random
import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc


def load_qwen3_model():
    """Load Qwen3 model using minimal number of GPUs"""
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    print("Loading Qwen3 model with concentrated GPU allocation...")
    
    # Clear GPU memory first
    print("🧹 Cleaning GPU memory before loading...")
    torch.cuda.empty_cache()
    gc.collect()
    
    # Check GPU memory status
    for i in range(torch.cuda.device_count()):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_cached = torch.cuda.memory_reserved(i) / 1024**3
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {mem_allocated:.1f}GB allocated, {mem_cached:.1f}GB cached, {mem_total:.1f}GB total")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Strategy 1: Use GPUs 5, 6, 7 (they show 0% usage)
    # Qwen3 has 48 layers, distribute across 3 GPUs
    if num_gpus >= 8:
        device_map = {
            "model.embed_tokens": 5,
        }
        
        # Distribute 48 layers across GPUs 5, 6, 7 (16 layers each)
        for i in range(48):
            if i < 16:
                gpu = 5
            elif i < 32:
                gpu = 6
            else:
                gpu = 7
            device_map[f"model.layers.{i}"] = gpu
        
        device_map.update({
            "model.norm": 7,
            "lm_head": 7
        })
        
        print("Using GPU strategy: concentrate on GPUs 5, 6, 7 (clean GPUs)")
        
    # Strategy 2: Use GPUs 1, 2, 3 if GPUs 5-7 not available
    elif num_gpus >= 4:
        device_map = {
            "model.embed_tokens": 1,
        }
        
        # Distribute 48 layers across GPUs 1, 2, 3
        for i in range(48):
            if i < 16:
                gpu = 1
            elif i < 32:
                gpu = 2
            else:
                gpu = 3
            device_map[f"model.layers.{i}"] = gpu
        
        device_map.update({
            "model.norm": 3,
            "lm_head": 3
        })
        
        print("Using GPU strategy: concentrate on GPUs 1, 2, 3")
    else:
        # Fallback: auto allocation
        device_map = "auto"
        print("Using fallback strategy: auto device mapping")
    
    try:
        # Load model with concentrated GPU allocation
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # Enable memory efficient attention if available
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        print("✅ Model loaded successfully with concentrated GPU allocation!")
        
        # Show actual device placement
        if hasattr(model, 'hf_device_map'):
            used_devices = set(model.hf_device_map.values())
            print(f"📊 Model distributed across devices: {used_devices}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"⚠️  Primary loading strategy failed: {e}")
        print("🔄 Trying CPU offload strategy...")
        
        # Fallback: CPU offload strategy
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="./offload",
                offload_state_dict=True,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
                
            print("✅ Model loaded with CPU offload strategy")
            return model, tokenizer
            
        except Exception as e2:
            print(f"❌ All loading strategies failed: {e2}")
            raise e2


def generate_incorrect_options(question, correct_answer, context, model, tokenizer, num_options=3):
    """Generate incorrect options using Qwen3 with memory optimization"""
    prompt = f"""Given this question and correct answer, generate {num_options} plausible but incorrect alternative answers.
The answers should be similar in format and style to the correct answer but factually wrong.

Question: {question}
Correct Answer: {correct_answer}
Context: {context[:500]}...

Generate {num_options} incorrect options that are:
1. Similar in format to the correct answer
2. Plausible but factually incorrect
3. Each on a separate line
4. Without numbering or bullet points

Incorrect options:"""

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Memory optimization
    with torch.no_grad():
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,  # Reduced for memory
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Clean up GPU memory immediately
        del model_inputs, generated_ids
        torch.cuda.empty_cache()
    
    # Parse the generated options
    options = []
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('Incorrect options:') and not line.startswith('Question:'):
            # Clean up any numbering or bullet points
            line = line.lstrip('123456789.- ')
            if line:
                options.append(line)
    
    # Return only the requested number of options
    return options[:num_options]


def count_total_questions(files):
    """Count total questions across all files"""
    total = 0
    print("Counting total questions...")
    for file in tqdm(files, desc="Scanning files"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
                total += count
                print(f"  {os.path.basename(file)}: {count} questions")
        except Exception as e:
            print(f"  Error reading {file}: {e}")
            continue
    
    print(f"Total questions to process: {total}")
    return total


def create_multiple_choice_question(item, model, tokenizer):
    """Convert a single item to multiple choice format"""
    question = item['question']
    correct_answer = item['answer'][0] if isinstance(item['answer'], list) else item['answer']
    context = item.get('context', '')
    
    # Generate 3 incorrect options
    incorrect_options = generate_incorrect_options(question, correct_answer, context, model, tokenizer)
    
    # Combine all options
    all_options = [correct_answer] + incorrect_options[:3]  # Ensure we have exactly 4 options
    
    # Pad with generic wrong answers if we don't have enough
    while len(all_options) < 4:
        all_options.append("None of the above")
    
    # Shuffle options and track correct answer position
    option_labels = ['A', 'B', 'C', 'D']
    shuffled_indices = list(range(4))
    random.shuffle(shuffled_indices)
    
    correct_index = shuffled_indices[0]  # Position of correct answer after shuffle
    correct_label = option_labels[correct_index]
    
    # Create shuffled options
    shuffled_options = [all_options[i] for i in shuffled_indices]
    
    # Format the question with options
    formatted_question = f"{question}\nChoose the correct one.\n"
    for label, option in zip(option_labels, shuffled_options):
        formatted_question += f"{label}. {option}\n"
    
    # Create the new item
    new_item = {
        'question': formatted_question.strip(),
        'answer': [correct_label],
        'correct_answer_text': correct_answer,
        'options': {
            'A': shuffled_options[0],
            'B': shuffled_options[1], 
            'C': shuffled_options[2],
            'D': shuffled_options[3]
        },
        'context': item.get('context', ''),
        'fact_context': item.get('fact_context', ''),
        'keyword': item.get('keyword', [])
    }
    
    return new_item


def process_jsonl_file(input_file, model, tokenizer, overall_pbar=None):
    """Process a single JSONL file to create multiple choice questions"""
    base_name = os.path.basename(input_file)
    print(f"\nProcessing {base_name}...")
    
    # Count lines first for progress bar
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    # Generate output filename (remove 'keyword', add 'options')
    if '_keyword.jsonl' in base_name:
        output_name = base_name.replace('_keyword.jsonl', '_options.jsonl')
    else:
        output_name = base_name.replace('.jsonl', '_options.jsonl')
    
    output_file = os.path.join(os.path.dirname(input_file), output_name)
    
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # File-specific progress bar
        file_pbar = tqdm(total=total_lines, 
                        desc=f"{base_name}", 
                        unit="questions",
                        position=1,
                        leave=False)
        
        for line_num, line in enumerate(infile, 1):
            try:
                if not line.strip():
                    continue
                    
                item = json.loads(line.strip())
                new_item = create_multiple_choice_question(item, model, tokenizer)
                outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                processed_count += 1
                
                # Update progress bars
                file_pbar.update(1)
                if overall_pbar:
                    overall_pbar.update(1)
                
                # Force garbage collection every 50 questions to save memory
                if processed_count % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                error_count += 1
                file_pbar.set_postfix({"errors": error_count})
                if overall_pbar:
                    overall_pbar.update(1)
                continue
        
        file_pbar.close()
    
    print(f"  ✓ {base_name}: {processed_count} questions, {error_count} errors → {output_name}")
    return output_file


def main():
    """Main function to process all keyword JSONL files"""
    print("=== Multiple Choice Question Generator ===")
    
    # Find all keyword JSONL files
    keyword_files = glob.glob("*_keyword.jsonl")
    
    if not keyword_files:
        print("❌ No *_keyword.jsonl files found in current directory")
        return
    
    print(f"📁 Found {len(keyword_files)} files to process:")
    for f in keyword_files:
        print(f"  - {f}")
    
    # Count total questions across all files
    total_questions = count_total_questions(keyword_files)
    
    if total_questions == 0:
        print("❌ No questions found to process")
        return
    
    # Load the model once
    try:
        model, tokenizer = load_qwen3_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Overall progress bar
    print(f"\n🚀 Starting processing of {total_questions} total questions...")
    overall_pbar = tqdm(total=total_questions, 
                       desc="Overall Progress", 
                       unit="questions",
                       position=0)
    
    # Process each file
    output_files = []
    successful_files = 0
    
    try:
        for input_file in keyword_files:
            try:
                output_file = process_jsonl_file(input_file, model, tokenizer, overall_pbar)
                output_files.append(output_file)
                successful_files += 1
                
                # Clean up memory between files
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error processing {input_file}: {e}")
                continue
    
    finally:
        overall_pbar.close()
        
        # Clean up model to free GPU memory
        try:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleaned up")
        except:
            pass
    
    # Final summary
    print(f"\n🎉 === Processing Complete ===")
    print(f"✅ Successfully processed: {successful_files}/{len(keyword_files)} files")
    print(f"📊 Generated {len(output_files)} multiple choice files:")
    
    for f in output_files:
        # Get file size for summary
        try:
            size_bytes = os.path.getsize(f)
            size_mb = size_bytes / (1024 * 1024)
            print(f"  ✓ {os.path.basename(f)} ({size_mb:.1f} MB)")
        except:
            print(f"  ✓ {os.path.basename(f)}")
    
    print(f"\n💾 All output files saved in current directory")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        # Clean up GPU memory
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        # Clean up GPU memory
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass