import os
import json
import torch
from datasets import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import transformers
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

# 1. Load the model with quantization
print("Loading Qwen/QVQ-72B-Preview model...")

# Configure quantization to save memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/QVQ-72B-Preview",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    # local_files_only=True,  # Commented out to allow downloading
    trust_remote_code=True
)

processor = Qwen2VLProcessor.from_pretrained(
    "Qwen/QVQ-72B-Preview",
    # local_files_only=True,  # Commented out to allow downloading
    trust_remote_code=True
)

# Ensure tokenizer has a pad token
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print("Set pad_token to eos_token")

# 2. Configure LoRA
print("Setting up LoRA configuration...")

# LoRA configuration for Qwen model
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Load your custom dataset
def load_custom_dataset(json_file_path, image_base_path=""):
    """
    Load your custom dataset from JSON file
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to the format expected by the training script
    processed_data = []
    for item in data:
        # Load the image
        image_path = os.path.join(image_base_path, item["image"][0])
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

# 4. Create conversation format for Qwen
def format_qwen_conversation(sample):
    """
    Format conversation for Qwen model
    """
    # Create the conversation
    question = sample["question"]
    answer = sample["answer"]
    solution = sample["solution"]
    
    # Qwen conversation format
    conversation = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": question}
            ]
        },
        {
            "role": "assistant", 
            "content": f"The answer is {answer}.\n\nExplanation: {solution}"
        }
    ]
    
    return conversation

# 5. Data processing function
def process_data(examples):
    """
    Process the data for training
    """
    conversations = []
    images = []
    
    for i in range(len(examples["question"])):
        sample = {
            "question": examples["question"][i],
            "answer": examples["answer"][i],
            "solution": examples["solution"][i],
            "image": examples["image"][i]
        }
        
        conversation = format_qwen_conversation(sample)
        
        # Apply chat template
        text = processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        conversations.append(text)
        images.append(sample["image"])
    
    # Process with the processor
    batch = processor(
        text=conversations,
        images=images,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt"
    )
    
    # Set labels for training (copy input_ids)
    batch["labels"] = batch["input_ids"].clone()
    
    return batch

# Process the dataset
print("Processing dataset...")
processed_dataset = dataset.map(
    process_data, 
    batched=True, 
    batch_size=4,
    remove_columns=dataset.column_names
)

# 6. Test the model before training
print("\n=== Testing model before training ===")
model.eval()
test_sample = raw_dataset[0]

# Create conversation for testing (exclude assistant response)
test_conversation = [
    {
        "role": "user", 
        "content": [
            {"type": "image", "image": test_sample["image"]},
            {"type": "text", "text": test_sample["question"]}
        ]
    }
]

# Prepare input
test_text = processor.apply_chat_template(
    test_conversation,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=[test_text],
    images=[test_sample["image"]],
    return_tensors="pt"
)

# Move inputs to the same device as model
if hasattr(model, 'device'):
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print("Input question:", test_sample["question"][:200] + "..." if len(test_sample["question"]) > 200 else test_sample["question"])
print("Expected answer:", test_sample["answer"])
print("\nModel response before training:")

with torch.no_grad():
    try:
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        response = processor.decode(output[0], skip_special_tokens=True)
        # Extract only the generated part
        if test_text in response:
            generated_part = response.replace(test_text, "").strip()
        else:
            generated_part = response
        print(generated_part)
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Continuing with training...")

# 7. Training setup
print("\n=== Setting up training ===")

# Custom data collator for vision-language model
class VisionLanguageDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        # Since we stored everything as lists, convert back to tensors
        batch_dict = {}
        
        # Handle different keys that might be present
        keys_to_process = ['input_ids', 'attention_mask', 'labels']
        tensor_keys = ['pixel_values', 'image_grid_thw']
        
        # Process regular keys (pad them)
        for key in keys_to_process:
            if key in batch[0]:
                # Find max length for padding
                max_len = max(len(item[key]) for item in batch)
                
                padded_items = []
                for item in batch:
                    values = item[key]
                    if key in ['input_ids', 'labels']:
                        pad_value = self.processor.tokenizer.pad_token_id if key == 'input_ids' else -100
                    else:  # attention_mask
                        pad_value = 0
                    
                    # Pad the sequence
                    padded = values + [pad_value] * (max_len - len(values))
                    padded_items.append(padded)
                
                batch_dict[key] = torch.tensor(padded_items, dtype=torch.long)
        
        # Process tensor keys (convert from lists to tensors)
        for key in tensor_keys:
            if key in batch[0]:
                tensors = [torch.tensor(item[key]) for item in batch]
                if key == 'pixel_values':
                    # Pixel values might need special handling
                    try:
                        batch_dict[key] = torch.stack(tensors)
                    except:
                        # If stacking fails, keep as list of tensors
                        batch_dict[key] = tensors
                else:
                    batch_dict[key] = torch.stack(tensors)
        
        return batch_dict

# Data collator
data_collator = VisionLanguageDataCollator(processor)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Increased to maintain effective batch size
    num_train_epochs=1,  # Reduced to 1 epoch for initial testing
    learning_rate=1e-4,  # Slightly reduced learning rate
    fp16=True,
    logging_steps=5,  # Reduced logging frequency
    save_steps=100,  # Increased save steps
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    warmup_steps=10,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    dataloader_drop_last=True,
    gradient_checkpointing=True,
    dataloader_num_workers=0,  # Reduce number of workers to avoid multiprocessing issues
    max_steps=50,  # Limit training steps for initial run
)

# Custom trainer class for vision-language models
class VisionLanguageTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for vision-language model
        """
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Create trainer
trainer = VisionLanguageTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
)

# 8. Start training
print("Starting training...")
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train the model
trainer.train()

# Memory usage after training
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# 9. Test the model after training
print("\n=== Testing model after training ===")
model.eval()

# Use the same test setup as before training
inputs = processor(
    text=[test_text],
    images=[test_sample["image"]],
    return_tensors="pt"
)

# Move inputs to the same device as model
if hasattr(model, 'device'):
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print("Model response after training:")
with torch.no_grad():
    try:
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        response = processor.decode(output[0], skip_special_tokens=True)
        # Extract only the generated part
        if test_text in response:
            generated_part = response.replace(test_text, "").strip()
        else:
            generated_part = response
        print(generated_part)
    except Exception as e:
        print(f"Error during generation: {e}")

# 10. Save the model
print("Saving model...")
model.save_pretrained("./qwen_lora_model")
processor.save_pretrained("./qwen_lora_model")

# Optional: Save merged model
# from peft import PeftModel
# base_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/QVQ-72B-Preview")
# merged_model = PeftModel.from_pretrained(base_model, "./qwen_lora_model")
# merged_model = merged_model.merge_and_unload()
# merged_model.save_pretrained("./qwen_merged_model")

print("Training completed and model saved!")