#!/usr/bin/env python
# FIXED: Model Parallelism with proper device handling for Llama-3.3-70B
# Usage: python fixed_model_parallel.py

import os
import sys
import json
import torch
import gc
import importlib.util
from typing import Optional

# CRITICAL: Environment setup for model parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def check_dependencies():
    required_packages = [
        "torch", "transformers", "datasets", "peft", 
        "bitsandbytes", "accelerate", "huggingface_hub", "sklearn"
    ]
    
    missing = []
    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing.append(package)
        except ModuleNotFoundError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)
    
    print("✅ All dependencies available")

check_dependencies()

# Core imports
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    logging, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.model_selection import train_test_split
from accelerate import Accelerator, DistributedDataParallelKwargs

# Memory management
def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_stats():
    """Print memory usage across all GPUs"""
    if torch.cuda.is_available():
        total_allocated = 0
        total_reserved = 0
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total_allocated += allocated
            total_reserved += reserved
            print(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        print(f"🔍 Total: {total_allocated:.2f}GB allocated, {total_reserved:.2f}GB reserved")

# Setup
logging.set_verbosity_error()
set_seed(42)

# Check GPU availability
if not torch.cuda.is_available():
    print("❌ No CUDA available")
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print(f"🎯 GPU Setup: {num_gpus} GPUs detected")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"   GPU {i}: {props.name} - {props.total_memory/1e9:.1f}GB")

# CONFIGURATION
CONFIG = {
    "model_name": "meta-llama/Llama-3.3-70B-Instruct",
    "dataset_path": "/home/mukesh/extramarks/final_data/final_combined_text_questions.json",
    "output_dir": "./llama-3.3-70b-model-parallel-fixed",
    "max_seq_length": 1024,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,  # Keep small for stability
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 32,  # Large accumulation for effective batch
    "learning_rate": 2e-5,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 0.5,
    "dataloader_num_workers": 2,
    "save_steps": 50,
    "eval_steps": 50,
    "logging_steps": 10,
    "save_total_limit": 2,
    "lora_r": 8,  # Conservative for stability
    "lora_alpha": 16,
    "lora_dropout": 0.05,
}

print("🔍 Initial Memory Usage:")
print_memory_stats()

# Initialize Accelerator for proper device handling
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    mixed_precision="bf16",
    kwargs_handlers=[ddp_kwargs]
)

print(f"📱 Accelerator device: {accelerator.device}")

# QUANTIZATION CONFIG
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LOAD TOKENIZER
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_name"],
    trust_remote_code=True,
    model_max_length=CONFIG["max_seq_length"],
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"✅ Tokenizer loaded")

# LOAD MODEL - FIXED APPROACH
print("🔄 Loading model with balanced device mapping...")

# Calculate memory distribution more carefully
memory_per_gpu = "18GiB"  # Conservative allocation per GPU
max_memory = {i: memory_per_gpu for i in range(num_gpus)}

try:
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        device_map="balanced",  # Use balanced instead of auto
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
        max_memory=max_memory,
        offload_folder="./offload",  # Offload to disk if needed
    )
    
    model.gradient_checkpointing_enable()
    
    print("✅ Model loaded with balanced device mapping!")
    
    # Print device mapping
    if hasattr(model, 'hf_device_map'):
        print("🗺️  Model device mapping:")
        device_counts = {}
        for layer, device in model.hf_device_map.items():
            if isinstance(device, int):
                device_counts[device] = device_counts.get(device, 0) + 1
        
        for device, count in device_counts.items():
            print(f"   GPU {device}: {count} layers")
    
    print("\n🔍 Memory usage after model loading:")
    print_memory_stats()
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# PREPROCESSING FUNCTION
def preprocess_batch(examples):
    """Preprocessing for model parallelism"""
    batch_size = len(examples["id"])
    results = {"input_ids": [], "attention_mask": []}
    
    for i in range(batch_size):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides detailed step-by-step solutions to questions."},
            {"role": "user", "content": examples["question"][i]},
            {"role": "assistant", "content": f"Here's my step-by-step solution:\n\n{examples['solution'][i]}\n\nFinal answer: {examples['answer'][i]}"}
        ]
        
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except:
            text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant that provides detailed step-by-step solutions to questions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{examples['question'][i]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHere's my step-by-step solution:\n\n{examples['solution'][i]}\n\nFinal answer: {examples['answer'][i]}<|eot_id|>"
        
        tokens = tokenizer(text, truncation=True, max_length=CONFIG["max_seq_length"], padding=False)
        results["input_ids"].append(tokens["input_ids"])
        results["attention_mask"].append(tokens["attention_mask"])
        
        del messages, text, tokens
        
        if (i + 1) % 100 == 0:
            clear_memory()
    
    clear_memory()
    return results

# LOAD DATASET
print("📚 Loading dataset...")
try:
    dataset = load_dataset("json", data_files=CONFIG["dataset_path"])
    raw_dataset = dataset["train"]
    
    required_fields = ["id", "question", "answer", "solution"]
    missing = [f for f in required_fields if f not in raw_dataset.features]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    
    print(f"📊 Dataset size: {len(raw_dataset)} samples")
    
    # Preprocess in smaller batches for stability
    print("🔄 Preprocessing dataset...")
    processed_dataset = raw_dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=50,  # Smaller batches
        remove_columns=raw_dataset.column_names,
        desc="Preprocessing"
    )
    
    clear_memory()
    
    # Split dataset
    dataset_size = len(processed_dataset)
    train_size = int(0.9 * dataset_size)
    
    train_indices = list(range(train_size))
    eval_indices = list(range(train_size, dataset_size))
    
    train_dataset = processed_dataset.select(train_indices)
    eval_dataset = processed_dataset.select(eval_indices)
    
    del processed_dataset, raw_dataset
    clear_memory()
    
    print(f"📈 Training samples: {len(train_dataset)}")
    print(f"📊 Evaluation samples: {len(eval_dataset)}")
    
except Exception as e:
    print(f"❌ Dataset error: {e}")
    sys.exit(1)

# SETUP LORA - FIXED FOR MODEL PARALLELISM
print("🔧 Setting up LoRA for model parallelism...")

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # Add device mapping for LoRA
    inference_mode=False,
)

# Prepare model for training with proper device handling
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, lora_config)

print("✅ LoRA setup complete")
model.print_trainable_parameters()

print("\n🔍 Memory usage after LoRA setup:")
print_memory_stats()

# CUSTOM DATA COLLATOR FOR MODEL PARALLELISM
class ModelParallelDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer, mlm)
        self.first_device = None
        
    def __call__(self, features):
        # Get the device of the first model parameter
        if self.first_device is None:
            try:
                # Find the first parameter to determine device
                for param in model.parameters():
                    self.first_device = param.device
                    break
                if self.first_device is None:
                    self.first_device = torch.device("cuda:0")
            except:
                self.first_device = torch.device("cuda:0")
        
        batch = super().__call__(features)
        
        # Move batch to the appropriate device
        if isinstance(batch, dict):
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.first_device)
        
        return batch

data_collator = ModelParallelDataCollator(tokenizer=tokenizer, mlm=False)

# TRAINING ARGUMENTS - OPTIMIZED FOR MODEL PARALLELISM
training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    
    # Optimization
    optim="adamw_torch",
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    max_grad_norm=CONFIG["max_grad_norm"],
    
    # Precision
    bf16=True,
    tf32=True,
    
    # Evaluation & Saving
    eval_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    load_best_model_at_end=False,  # Disable for model parallelism
    metric_for_best_model="eval_loss",
    
    # Logging
    logging_steps=CONFIG["logging_steps"],
    logging_dir=f"{CONFIG['output_dir']}/logs",
    report_to="none",
    
    # Data loading - Conservative settings
    dataloader_num_workers=0,  # Single worker for stability
    dataloader_pin_memory=False,
    dataloader_drop_last=True,
    group_by_length=True,
    
    # Memory optimization
    remove_unused_columns=False,
    auto_find_batch_size=False,
    save_only_model=True,
    gradient_checkpointing=True,
    
    # Disable problematic features for model parallelism
    prediction_loss_only=True,
    include_inputs_for_metrics=False,
)

# CUSTOM TRAINER FOR MODEL PARALLELISM
class ModelParallelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def create_optimizer(self):
        """Create optimizer with model parallelism support"""
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to handle device placement properly"""
        
        # Ensure inputs are on the right device
        if "input_ids" in inputs:
            # Find the device of the embedding layer
            embed_device = None
            try:
                if hasattr(model, 'base_model'):
                    embed_device = next(model.base_model.model.model.embed_tokens.parameters()).device
                else:
                    embed_device = next(model.model.embed_tokens.parameters()).device
            except:
                embed_device = torch.device("cuda:0")
            
            # Move inputs to embedding device
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(embed_device)
        
        # Standard forward pass
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        outputs = model(**inputs)
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
            
        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
        return (loss, outputs) if return_outputs else loss

# Initialize trainer
trainer = ModelParallelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# TRAINING INFO
effective_batch_size = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
steps_per_epoch = len(train_dataset) // effective_batch_size
total_steps = steps_per_epoch * CONFIG["num_train_epochs"]

print(f"\n🎯 TRAINING CONFIGURATION:")
print(f"   Model Distribution: Balanced across {num_gpus} GPUs")
print(f"   Effective batch size: {effective_batch_size}")
print(f"   Steps per epoch: {steps_per_epoch}")
print(f"   Total steps: {total_steps}")
print(f"   Estimated time: {total_steps * 2.0 / 60:.1f} minutes")

print("\n🔍 Memory usage before training:")
print_memory_stats()

# START TRAINING
try:
    print(f"\n🚀 STARTING FIXED MODEL PARALLEL TRAINING")
    
    trainer.train()
    
    clear_memory()
    
    print("\n💾 Saving final model...")
    final_path = f"{CONFIG['output_dir']}/final_model"
    
    # Save with special handling for model parallelism
    trainer.model.save_pretrained(
        final_path,
        safe_serialization=True,
        max_shard_size="2GB"  # Smaller shards for better handling
    )
    tokenizer.save_pretrained(final_path)
    
    print(f"✅ Model saved to: {final_path}")
    
    print("\n🔍 Final memory usage:")
    print_memory_stats()
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Clean up on error
    if os.path.exists("./offload"):
        import shutil
        shutil.rmtree("./offload")

clear_memory()

print(f"\n🎉 MODEL PARALLEL TRAINING COMPLETED!")
print(f"📁 Output directory: {CONFIG['output_dir']}")

# Clean up offload folder
if os.path.exists("./offload"):
    import shutil
    shutil.rmtree("./offload")
    print("🧹 Cleaned up offload folder")