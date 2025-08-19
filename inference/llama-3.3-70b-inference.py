import os
import json
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer
)
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings("ignore")

class Llama33InferenceEngine:
    def __init__(self, base_model_path, lora_model_path, use_4bit=True):
        """
        Initialize the inference engine for Llama-3.3-70B-Instruct
        
        Args:
            base_model_path: Path to base model (e.g., "meta-llama/Llama-3.3-70B-Instruct")
            lora_model_path: Path to your LoRA adapters (e.g., "./final_model")
            use_4bit: Whether to use 4-bit quantization
        """
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.use_4bit = use_4bit
        
        print("Loading Llama-3.3-70B-Instruct model...")
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
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            local_files_only=True,  # Use cached version
            trust_remote_code=True
        )
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set pad_token to eos_token")
        
        # Load base model
        print("Loading base model...")
        self.base_model = LlamaForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True
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
        
    def inference_single(self, prompt, max_new_tokens=4096, temperature=0.7, use_chat_template=True):
        """
        Run inference on a single prompt
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_chat_template: Whether to format as chat conversation
            
        Returns:
            Generated response
        """
        # Format prompt using chat template if requested
        if use_chat_template:
            messages = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        # Move to device
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1,
                top_p=0.9
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the input prompt)
        if formatted_prompt in generated_text:
            response = generated_text.replace(formatted_prompt, "").strip()
        else:
            response = generated_text
            
        return response
    
    def inference_streaming(self, prompt, max_new_tokens=4096, temperature=0.1, 
                        use_chat_template=True, output_format="plain_math", 
                        random_seed=42, consistent_mode=True):
        """
        Run inference with streaming output (real-time token generation)
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more consistent)
            use_chat_template: Whether to format as chat conversation
            output_format: "structured", "plain", "brief", "plain_math"
            random_seed: Seed for reproducible results
            consistent_mode: Enable consistent output settings
        """
        
        # Format instructions for different output types
        format_instructions = {
            "structured": "Please provide a detailed step-by-step solution with clear headings and mathematical expressions in LaTeX format.",
            "plain": "Please provide a concise answer in plain text without special formatting or LaTeX.",
            "brief": "Please provide only the final answer with minimal explanation.",
            "plain_math": "Please provide a detailed step-by-step solution with clear headings. Write all mathematical expressions using simple text notation (like v = ω × L × sin(θ)) instead of LaTeX. Use symbols like ×, ÷, √, π, θ, ω, etc. directly in plain text.",
            "auto": ""  # Let model decide naturally
        }
        
        # Enhanced prompt with format instruction
        if output_format in format_instructions and format_instructions[output_format]:
            enhanced_prompt = f"{prompt}\n\n{format_instructions[output_format]}"
        else:
            enhanced_prompt = prompt
        
        # Format prompt with system message for consistency
        if use_chat_template:
            if output_format == "structured":
                system_msg = "You are a helpful physics tutor. Always provide step-by-step solutions with clear headings, mathematical expressions in LaTeX format, and detailed explanations."
            elif output_format == "plain":
                system_msg = "You are a helpful assistant. Provide clear, concise answers in plain text without special formatting."
            elif output_format == "plain_math":
                system_msg = "You are a helpful physics tutor. Provide step-by-step solutions with clear headings. Write all mathematical expressions using simple text notation (like F = m × a, v = √(2gh), etc.) instead of LaTeX. Use standard mathematical symbols (×, ÷, √, π, θ, ω, Δ, etc.) directly in plain text. Never use LaTeX formatting like \\[ \\], \\frac{}{}, or similar."
            elif output_format == "brief":
                system_msg = "You are a helpful assistant. Provide brief, direct answers."
            else:
                system_msg = "You are a helpful assistant."
                
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": enhanced_prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = enhanced_prompt
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Set random seed for reproducibility
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Create text streamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        print(f"User: {prompt}")
        print(f"Assistant: ", end="")
        
        # Adjust generation parameters based on consistency mode
        if consistent_mode:
            generation_params = {
                "max_new_tokens": max_new_tokens,
                "temperature": min(temperature, 0.3),  # Cap temperature for consistency
                "do_sample": True if temperature > 0 else False,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
                "use_cache": True,
                "repetition_penalty": 1.05,  # Lower repetition penalty
                "top_p": 0.95,  # Higher top_p for more consistent choices
                "top_k": 50,    # Add top_k for additional consistency
            }
        else:
            # Original parameters for more creative/varied output
            generation_params = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True if temperature > 0 else False,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
                "use_cache": True,
                "repetition_penalty": 1.1,
                "top_p": 0.9
            }
        
        # Generate with streaming
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                **generation_params
            )

    def inference_streaming_plain_math(self, prompt, max_new_tokens=4096, 
                                     use_chat_template=True):
        """
        Streaming inference specifically for plain text mathematical expressions
        """
        return self.inference_streaming(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for consistency
            use_chat_template=use_chat_template,
            output_format="plain_math",
            random_seed=42,
            consistent_mode=True
        )

    def inference_streaming_deterministic(self, prompt, max_new_tokens=4096, 
                                        use_chat_template=True, output_format="structured"):
        """
        Completely deterministic streaming (most consistent results)
        """
        return self.inference_streaming(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # No randomness
            use_chat_template=use_chat_template,
            output_format=output_format,
            random_seed=42,
            consistent_mode=True
        )

    def inference_streaming_creative(self, prompt, max_new_tokens=4096, 
                                temperature=0.8, use_chat_template=True):
        """
        More creative/varied streaming output
        """
        return self.inference_streaming(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_chat_template=use_chat_template,
            output_format="auto",
            random_seed=None,  # Different results each time
            consistent_mode=False
        )
    
    def inference_batch(self, prompts, max_new_tokens=4096, temperature=0.7, use_chat_template=True):
        """
        Run inference on a batch of prompts
        
        Args:
            prompts: List of text prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_chat_template: Whether to format as chat conversation
            
        Returns:
            List of generated responses
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            
            response = self.inference_single(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_chat_template=use_chat_template
            )
            
            results.append({
                'prompt': prompt,
                'response': response
            })
        
        return results
    
    def evaluate_on_dataset(self, json_file_path, prompt_key="question", answer_key="answer", num_samples=10):
        """
        Evaluate the model on a dataset
        
        Args:
            json_file_path: Path to your JSON dataset
            prompt_key: Key in JSON for the input prompt
            answer_key: Key in JSON for expected answer
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
        
        results = []
        correct = 0
        
        for i, item in enumerate(data):
            print(f"Processing item {i+1}/{len(data)}")
            
            prompt = item.get(prompt_key, "")
            expected_answer = item.get(answer_key, "")
            
            response = self.inference_single(
                prompt,
                max_new_tokens=256,
                temperature=0.1  # Lower temperature for consistent evaluation
            )
            
            # Simple exact match check (you may want to customize this)
            is_correct = expected_answer.upper() in response.upper()
            if is_correct:
                correct += 1
            
            results.append({
                'prompt': prompt,
                'expected_answer': expected_answer,
                'response': response,
                'correct': is_correct
            })
        
        accuracy = correct / len(results) * 100
        
        print(f"\nEvaluation Results:")
        print(f"Total samples: {len(results)}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        return results, accuracy
    
    def chat_mode(self):
        """
        Interactive chat mode
        """
        print("\n=== Interactive Chat Mode ===")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'clear' to clear conversation history")
        print("-" * 40)
        
        conversation_history = []
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = []
                print("Conversation history cleared!")
                continue
            
            if not user_input:
                continue
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Format conversation
            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            
            print("Assistant: ", end="")
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                    use_cache=True,
                    repetition_penalty=1.1,
                    top_p=0.9
                )
            
            # Extract assistant response and add to history
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = full_response.replace(formatted_prompt, "").strip()
            conversation_history.append({"role": "assistant", "content": assistant_response})

# Utility functions
def merge_and_save_model(base_model_path, lora_path, output_path):
    """
    Merge LoRA adapters with base model and save
    """
    print("Loading base model...")
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging adapters...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Merged model saved successfully!")

# Example usage
if __name__ == "__main__":
    # Initialize the inference engine
    print("=== Loading Llama-3.3-70B-Instruct with LoRA adapters ===")
    
    inferencer = Llama33InferenceEngine(
        base_model_path="meta-llama/Llama-3.3-70B-Instruct",
        lora_model_path="/home/mukesh/extramarks/fine_tuning/llama-3.3-70b-model-parallel-fixed/final_model",
        use_4bit=True
    )
    
    # Example: Plain text mathematical expressions
    print("\n=== Plain Text Math Inference Example ===")
    prompt = "A simple pendulum with bob of mass m and conducting wire of length L swings under gravity through an angle 2θ. The earth's magnetic field component in the direction perpendicular to swing is B. Calculate the maximum induced emf across the pendulum?"
    prompt = """The angle of elevation of the top P
 of a tower from the feet of one person standing due South of the tower is 45deg
 and from the feet of another person standing due west of the tower is 30deg
. If the height of the tower is 5 meters, then the distance (in meters) between the two persons is equal to"""
    # Use the new plain_math format
    inferencer.inference_streaming_plain_math(prompt)
    
    # Alternative: Use the general method with plain_math format
    # inferencer.inference_streaming(
    #     prompt, 
    #     output_format="plain_math",
    #     temperature=0.1,
    #     consistent_mode=True
    # )