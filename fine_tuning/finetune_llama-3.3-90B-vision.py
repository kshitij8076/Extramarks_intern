import os
import json
from unsloth import FastVisionModel
import torch
from datasets import Dataset
from transformers import TextStreamer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from PIL import Image

# 1. Load the model
model, tokenizer = FastVisionModel.from_pretrained(
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    load_in_4bit = True,  # You might want to use load_in_8bit = True for 90B model
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules      = True,
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 2. Load your custom dataset
def load_custom_dataset(json_file_path, image_base_path=""):
    """
    Load your custom dataset from JSON file
    
    Args:
        json_file_path: Path to your JSON file containing the dataset
        image_base_path: Base path to prepend to image paths if needed
    """
    with open(json_file_path, 'r') as f:
        # If your data is a single JSON object, use json.load(f)
        # If your data is JSONL (one JSON per line), use:
        # data = [json.loads(line) for line in f]
        data = json.load(f)
    
    # Convert to the format expected by the training script
    processed_data = []
    for item in data:
        # Load the image
        image_path = os.path.join(image_base_path, item["image"][0])  # Taking first image
        try:
            image = Image.open(image_path).convert("RGB")
            processed_data.append({
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "solution": item["solution"],
                "image": image
            })
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
    
    return processed_data

# Load your dataset - MODIFY THESE PATHS
dataset_json_path = "/home/mukesh/extramarks/final_data/final_combined_image_questions_2.json"  # Path to your JSON file
image_base_path = "/home/mukesh/extramarks/final_data"  # Base path for images if needed
raw_dataset = load_custom_dataset(dataset_json_path, image_base_path)

# Convert to HuggingFace Dataset format
dataset = Dataset.from_list(raw_dataset)

# 3. Convert dataset to conversation format (FIXED VERSION)
def convert_to_conversation(sample):
    """
    Convert your dataset format to conversation format
    Fixed to work properly with vision models
    """
    # Create the user message with image first, then text
    user_content = [
        {"type": "image", "image": sample["image"]},
        {"type": "text", "text": sample["question"]}
    ]
    
    # Create assistant response with answer and solution
    assistant_text = f"The answer is {sample['answer']}.\n\nExplanation: {sample['solution']}"
    
    conversation = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": assistant_text}
            ]
        }
    ]
    return {"messages": conversation}

# Convert dataset
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# 4. Before training - test the model
FastVisionModel.for_inference(model)
test_sample = dataset[0]
test_image = test_sample["image"]
test_question = test_sample["question"]

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": test_question}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

print("\nBefore training:\n")
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)

# 5. Training (FIXED CONFIGURATION)
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,  # Keep small for 90B model
        gradient_accumulation_steps=8,  # Maintain effective batch size
        warmup_steps=5,
        max_steps=50,  
        learning_rate=1e-4,  
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        # CRITICAL FIX: Increase max_seq_length to avoid truncation issues
        max_seq_length=4096,  # Increased from 2048
        save_steps=25,  
        save_total_limit=2,
        # Additional fixes for vision models
        dataloader_pin_memory=False,  # Can help with memory issues
        group_by_length=False,  # Disable grouping which can cause issues with vision
    ),
)

# Memory monitoring
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Start training with error handling
try:
    trainer_stats = trainer.train()
    
    # Training statistics
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
except Exception as e:
    print(f"Training failed with error: {e}")
    print("Try reducing batch size or sequence length further")
    raise

# 6. After training - test the model
print("\nAfter training:\n")
FastVisionModel.for_inference(model)

# Test with the same sample
inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)

# 7. Save the model
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Optional: Save merged model (uncomment and modify as needed)
# model.save_pretrained_merged("your_username/Llama-3.2-90B-Vision-ScienceQA", tokenizer)
# model.push_to_hub_merged("your_username/Llama-3.2-90B-Vision-ScienceQA", tokenizer, 
#                         save_method="merged_16bit", token=os.environ.get("HF_TOKEN"))

print("Training completed and model saved!")