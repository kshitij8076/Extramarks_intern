import os
import json
import torch
from transformers import (
    MllamaForConditionalGeneration,  # Changed from LlavaNextForConditionalGeneration
    MllamaProcessor,                 # Changed from LlavaNextProcessor
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

class LlamaVisionInference:
    def __init__(self, base_model_path, lora_model_path, use_4bit=True):
        """
        Initialize the inference class
        
        Args:
            base_model_path: Path to base model (e.g., "meta-llama/Llama-3.2-90B-Vision-Instruct")
            lora_model_path: Path to your LoRA adapters (e.g., "./lora_model")
            use_4bit: Whether to use 4-bit quantization
        """
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.use_4bit = use_4bit
        
        print("Loading model for inference...")
        self._load_model()
        
    def _load_model(self):
        """Load the base model and LoRA adapters"""
        
        # Configure quantization if needed
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load base model
        print("Loading base model...")
        self.base_model = MllamaForConditionalGeneration.from_pretrained(  # Changed
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,  # Use cached version
        )
        
        # Load processor
        self.processor = MllamaProcessor.from_pretrained(  # Changed
            self.base_model_path,
            local_files_only=True,
        )
        
        # Load LoRA adapters
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.lora_model_path,
            torch_dtype=torch.float16,
        )
        
        # Set to evaluation mode
        self.model.eval()
        print("Model loaded successfully!")
        
    def inference_single(self, image_path, question, max_new_tokens=256, temperature=0.7):
        """
        Run inference on a single image-question pair
        
        Args:
            image_path: Path to the image file
            question: Question about the image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error loading image: {e}"
        
        # Create messages format for Llama 3.2 Vision
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Apply chat template
        input_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=input_text,
            return_tensors="pt"
        )
        
        # Move to device
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the input prompt)
        if input_text in generated_text:
            response = generated_text.replace(input_text, "").strip()
        else:
            response = generated_text
            
        return response
    
    def inference_batch(self, data_list, max_new_tokens=256, temperature=0.7):
        """
        Run inference on a batch of image-question pairs
        
        Args:
            data_list: List of dicts with 'image_path' and 'question' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated responses
        """
        results = []
        
        for i, item in enumerate(data_list):
            print(f"Processing item {i+1}/{len(data_list)}")
            
            response = self.inference_single(
                item['image_path'],
                item['question'],
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            results.append({
                'image_path': item['image_path'],
                'question': item['question'],
                'response': response,
                'expected_answer': item.get('expected_answer', 'N/A'),
                'solution': item.get('solution', 'N/A')
            })
        
        return results
    
    def evaluate_on_dataset(self, json_file_path, image_base_path="", num_samples=10):
        """
        Evaluate the model on your dataset
        
        Args:
            json_file_path: Path to your JSON dataset
            image_base_path: Base path for images
            num_samples: Number of samples to evaluate (set to -1 for all)
            
        Returns:
            Evaluation results
        """
        print(f"Loading dataset from {json_file_path}")
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Take subset if specified
        if num_samples > 0:
            data = data[:num_samples]
        
        # Prepare data for inference
        inference_data = []
        for item in data:
            image_path = os.path.join(image_base_path, item["image"][0])
            inference_data.append({
                'image_path': image_path,
                'question': item['question'],
                'expected_answer': item['answer'],
                'solution': item['solution']
            })
        
        # Run inference
        results = self.inference_batch(inference_data)
        
        # Calculate accuracy (simple exact match)
        correct = 0
        for result in results:
            if result['expected_answer'].upper() in result['response'].upper():
                correct += 1
        
        accuracy = correct / len(results) * 100
        
        print(f"\nEvaluation Results:")
        print(f"Total samples: {len(results)}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        return results, accuracy

# Method 1: Load from LoRA adapters
def load_from_lora_adapters():
    """Example: Load model with LoRA adapters"""
    
    # Initialize inference class
    inferencer = LlamaVisionInference(
        base_model_path="meta-llama/Llama-3.2-90B-Vision-Instruct",
        lora_model_path="/home/mukesh/extramarks/fine_tuning/llama-3.2-90b-vision-adapters/lora_model",
        use_4bit=True
    )
    
    return inferencer

# Method 2: Load from checkpoint
def load_from_checkpoint(checkpoint_path):
    """Example: Load from a specific checkpoint"""
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model from checkpoint
    model = MllamaForConditionalGeneration.from_pretrained(  # Changed
        checkpoint_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    processor = MllamaProcessor.from_pretrained(checkpoint_path)  # Changed
    
    return model, processor

# Method 3: Merge LoRA and save (optional - for faster inference)
def merge_and_save_model(base_model_path, lora_path, output_path):
    """
    Merge LoRA adapters with base model and save
    This creates a single model without needing LoRA loading
    """
    print("Loading base model...")
    base_model = MllamaForConditionalGeneration.from_pretrained(  # Changed
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging adapters...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Also save processor
    processor = MllamaProcessor.from_pretrained(base_model_path)  # Changed
    processor.save_pretrained(output_path)
    
    print("Merged model saved successfully!")

# Example usage
if __name__ == "__main__":
    # Method 1: Use LoRA adapters (Recommended)
    print("=== Loading model with LoRA adapters ===")
    inferencer = load_from_lora_adapters()
    
    # Single inference example
    print("\n=== Single Inference Example ===")
    # image_path = "path/to/your/test/image.jpg"  # Replace with actual path
    # question = "What can Edwin and Brenda trade to each get what they want?"
    
    img_url = "/home/mukesh/extramarks/phy.jpg"
    question2 = "A simple pendulum with bob of mass m and conducting wire of length L swings under gravity through an angle 2θ​. The earth's magnetic field component in the direction perpendicular to swing is B.Calculate the maximum induced emf across the pendulum?"
    # Uncomment to test single inference
    response = inferencer.inference_single(img_url, question2)
    # print(f"Question: {question}")
    # print(f"Response: {response}")
    # Evaluate on dataset
    print("\n=== Dataset Evaluation ===")
    json_file = "/home/mukesh/extramarks/final_data/final_combined_image_questions_2.json"
    image_base = "/home/mukesh/extramarks/final_data"
    
    # Evaluate on first 5 samples (change as needed)
    results, accuracy = inferencer.evaluate_on_dataset(
        json_file, 
        image_base, 
        num_samples=5
    )
    
    # Print detailed results
    print("\n=== Detailed Results ===")
    for i, result in enumerate(results):
        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {result['question'][:100]}...")
        print(f"Expected: {result['expected_answer']}")
        print(f"Generated: {result['response']}")
        print(f"Match: {'✓' if result['expected_answer'].upper() in result['response'].upper() else '✗'}")
    
    # Optional: Save results to file
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to inference_results.json")
    
    # Optional: Merge and save model for faster future inference
    print("\n=== Optional: Merge and Save Model ===")
    # Uncomment to merge and save
    # merge_and_save_model(
    #     base_model_path="meta-llama/Llama-3.2-90B-Vision-Instruct",
    #     lora_path="./lora_model",
    #     output_path="./merged_model"
    # )