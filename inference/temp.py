import streamlit as st
import os
import json
import torch
import warnings
from PIL import Image
import tempfile
import time
from io import BytesIO
import sys
import requests
import base64
import re

# Try to import OpenAI for image generation
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Compatibility fixes for Python 3.13
import asyncio
try:
    # Try to set event loop policy for compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    elif hasattr(asyncio, 'set_event_loop_policy'):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except:
    pass

# Import necessary libraries for different models
try:
    from transformers import (
        MllamaForConditionalGeneration, MllamaProcessor,
        LlamaForCausalLM, AutoTokenizer,
        AutoProcessor,
        BitsAndBytesConfig, TextStreamer
    )
    from peft import PeftModel
    
    # Try to import Llama4 - might not be available
    try:
        from transformers import Llama4ForConditionalGeneration
        LLAMA4_AVAILABLE = True
    except ImportError:
        LLAMA4_AVAILABLE = False
    
    # Try to import Qwen2VL
    try:
        from transformers import Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        QWEN2VL_AVAILABLE = True
    except ImportError:
        QWEN2VL_AVAILABLE = False
        
except ImportError as e:
    st.error(f"Error importing required libraries: {e}")
    st.stop()

warnings.filterwarnings("ignore")

class ModelManager:
    """Manages all different models and their inference"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        
        # Predefined LoRA paths for fine-tuned models
        self.lora_paths = {
            "llama_vision": "/home/mukesh/extramarks/fine_tuning/llama-3.2-90b-vision-adapters/lora_model",
            "llama33": "/home/mukesh/extramarks/fine_tuning/llama-3.3-70b-model-parallel-fixed/final_model"
        }
        
    def load_llama_vision(self, base_model_path, use_4bit=True):
        """Load Llama Vision model with automatic LoRA loading"""
        try:
            st.info("🔄 Loading Llama Vision model...")
            
            # Configure quantization
            bnb_config = None
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load base model
            st.info("📦 Loading base model...")
            base_model = MllamaForConditionalGeneration.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            
            processor = MllamaProcessor.from_pretrained(
                base_model_path,
                local_files_only=True,
            )
            
            # Load LoRA if path exists
            lora_path = self.lora_paths.get("llama_vision")
            if lora_path and os.path.exists(lora_path):
                st.info("🔧 Loading LoRA adapters...")
                model = PeftModel.from_pretrained(
                    base_model,
                    lora_path,
                    torch_dtype=torch.float16,
                )
                st.success("✅ LoRA adapters loaded successfully!")
            else:
                model = base_model
                st.info("ℹ️ Using base model (no LoRA adapters found)")
            
            model.eval()
            
            self.models['llama_vision'] = {
                'model': model,
                'processor': processor,
                'type': 'multimodal'
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading Llama Vision model: {str(e)}")
            return False
    
    def load_llama33(self, base_model_path, use_4bit=True):
        """Load Llama 3.3 text model with automatic LoRA loading"""
        try:
            st.info("🔄 Loading Llama 3.3 model...")
            
            bnb_config = None
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            
            st.info("📦 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            st.info("📦 Loading base model...")
            base_model = LlamaForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # Load LoRA if path exists
            lora_path = self.lora_paths.get("llama33")
            if lora_path and os.path.exists(lora_path):
                st.info("🔧 Loading LoRA adapters...")
                model = PeftModel.from_pretrained(
                    base_model,
                    lora_path,
                    torch_dtype=torch.float16,
                )
                st.success("✅ LoRA adapters loaded successfully!")
            else:
                model = base_model
                st.info("ℹ️ Using base model (no LoRA adapters found)")
            
            model.eval()
            
            self.models['llama33'] = {
                'model': model,
                'tokenizer': tokenizer,
                'type': 'text_only'
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading Llama 3.3 model: {str(e)}")
            return False
    
    def load_llama4_scout(self):
        """Load Llama 4 Scout model"""
        if not LLAMA4_AVAILABLE:
            st.error("❌ Llama4ForConditionalGeneration is not available in your transformers version")
            return False
            
        try:
            st.info("🔄 Loading Llama 4 Scout model...")
            model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            processor = AutoProcessor.from_pretrained(model_id)
            model = Llama4ForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="sdpa",
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            
            self.models['llama4_scout'] = {
                'model': model,
                'processor': processor,
                'tokenizer': tokenizer,
                'type': 'multimodal'
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading Llama 4 Scout model: {str(e)}")
            return False
    
    def load_qwen2vl(self):
        """Load Qwen2VL model"""
        if not QWEN2VL_AVAILABLE:
            st.error("❌ Qwen2VL is not available. Please install with: pip install qwen-vl-utils")
            return False
            
        try:
            st.info("🔄 Loading Qwen2VL model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/QVQ-72B-Preview", 
                torch_dtype="auto", 
                device_map="auto"
            )
            
            processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview")
            
            self.models['qwen2vl'] = {
                'model': model,
                'processor': processor,
                'type': 'multimodal'
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading Qwen2VL model: {str(e)}")
            return False
    
    def inference_llama_vision(self, question, image_path=None, max_tokens=1024, temperature=0.7):
        """Run inference with Llama Vision model"""
        model_data = self.models['llama_vision']
        model = model_data['model']
        processor = model_data['processor']
        
        # Enhanced system prompt for structured output
        system_content = (
            "You are a mathematics expert providing clear, detailed step-by-step solutions. "
            "Always structure your response as follows:\n\n"
            "**Problem Understanding:**\n"
            "- Clearly state what is given\n"
            "- Identify what needs to be found\n\n"
            "**Solution Approach:**\n"
            "- Explain the method or formula to be used\n"
            "- Break down the problem into steps\n\n"
            "**Step-by-Step Calculation:**\n"
            "- Show each calculation clearly\n"
            "- Explain the reasoning for each step\n"
            "- Use proper mathematical notation\n\n"
            "**Final Answer:**\n"
            "- State the final result clearly\n"
            "- Include appropriate units\n\n"
            "If an image is provided, analyze it carefully and extract all relevant information before solving."
        )
        
        # Create messages with system prompt
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_content}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"} if image_path else {},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Remove empty image content if no image
        if not image_path:
            messages[1]["content"] = [{"type": "text", "text": question}]
        
        input_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        if image_path:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                images=image,
                text=input_text,
                return_tensors="pt"
            )
        else:
            inputs = processor(
                text=input_text,
                return_tensors="pt"
            )
        
        # Move to device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        if input_text in generated_text:
            response = generated_text.replace(input_text, "").strip()
        else:
            response = generated_text
            
        return response
    
    def inference_llama33(self, question, max_tokens=1024, temperature=0.7):
        """Run inference with Llama 3.3 model"""
        model_data = self.models['llama33']
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        # Enhanced system prompt for structured output
        system_content = (
            "You are a mathematics expert providing clear, detailed step-by-step solutions. "
            "Always structure your response in the following format:\n\n"
            "## Problem Understanding\n"
            "- Given: [List all given information]\n"
            "- Find: [What needs to be calculated]\n\n"
            "## Solution Approach\n"
            "- Method: [Explain the approach/formula to use]\n"
            "- Strategy: [Break down the solution strategy]\n\n"
            "## Step-by-Step Solution\n"
            "**Step 1:** [Description]\n"
            "[Calculation with clear explanation]\n\n"
            "**Step 2:** [Description]\n"
            "[Calculation with clear explanation]\n\n"
            "[Continue for all steps...]\n\n"
            "## Final Answer\n"
            "**Result:** [Final numerical answer with units]\n\n"
            "Use clear mathematical notation and explain your reasoning at each step. "
            "Make sure all calculations are clearly shown and easy to follow."
        )
        
        # Format with chat template including system message
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1,
                top_p=0.9
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if formatted_prompt in generated_text:
            response = generated_text.replace(formatted_prompt, "").strip()
        else:
            response = generated_text
            
        return response
    
    def inference_llama4_scout(self, question, image_path=None, max_tokens=512, temperature=0.7, request_diagram=False):
        """Run inference with Llama 4 Scout model with optional diagram generation"""
        model_data = self.models['llama4_scout']
        model = model_data['model']
        processor = model_data['processor']
        
        # Enhanced system prompt for image generation
        system_content = (
            "You are a mathematics expert. The user will provide a math question consisting of an image and/or text description. "
            "Carefully analyze any provided image to extract necessary mathematical information (lengths, angles, geometry, diagrams, etc.). "
            "Then, use this information along with the text to solve the math question step-by-step. Show all workings clearly and compute the final answer. "
            "Use proper LaTeX formatting for mathematical expressions when appropriate.\n\n"
            
            "IMPORTANT: If the solution would benefit from a visual diagram or if the user specifically requests an output diagram, "
            "you MUST include a section at the end of your response with the following format:\n"
            "**IMAGE_GENERATION_NEEDED**\n"
            "PROMPT: [Detailed description for generating a clear, educational diagram that would help visualize the solution. "
            "Include specific measurements, labels, geometric shapes, and styling instructions for a textbook-style diagram.]\n"
            "**END_IMAGE_GENERATION**\n\n"
            
            "The image prompt should be detailed enough for an AI image generator to create an accurate mathematical diagram."
        )
        
        # Create messages
        user_content = []
        if image_path:
            user_content.append({"type": "image", "url": image_path})
        
        # Modify question to request diagram if needed
        if request_diagram:
            question += "\n\nPlease also provide an output diagram to visualize the solution."
            
        user_content.append({"type": "text", "text": question})
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_content}]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )
        
        response = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )[0]
        
        return response
    
    def inference_qwen2vl(self, question, image_path=None, max_tokens=512, temperature=0.3):
        """Run inference with Qwen2VL model"""
        model_data = self.models['qwen2vl']
        model = model_data['model']
        processor = model_data['processor']
        
        # Create messages
        user_content = []
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        user_content.append({"type": "text", "text": question})
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a mathematics expert providing clear, step-by-step solutions."}]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""
    
    def extract_image_prompt(self, llama_response):
        """Extract image generation prompt from LLaMA response"""
        # Look for the image generation markers
        pattern = r'\*\*IMAGE_GENERATION_NEEDED\*\*\n(.*?)\*\*END_IMAGE_GENERATION\*\*'
        match = re.search(pattern, llama_response, re.DOTALL)
        
        if match:
            image_section = match.group(1).strip()
            # Extract the prompt part
            prompt_match = re.search(r'PROMPT:\s*(.*)', image_section, re.DOTALL)
            if prompt_match:
                return prompt_match.group(1).strip()
        
        return None
    
    def generate_image_with_dalle(self, prompt, api_key):
        """Generate image using DALL-E 3"""
        if not OPENAI_AVAILABLE:
            return None, "OpenAI library not available. Please install with: pip install openai"
            
        try:
            # Set the API key
            openai.api_key = api_key
            
            # Generate image
            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard"
            )
            
            image_url = response.data[0].url
            return image_url, None
            
        except Exception as e:
            return None, f"Error generating image: {str(e)}"
    
    def download_and_convert_image(self, image_url):
        """Download image from URL and convert to PIL Image"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image, None
        except Exception as e:
            return None, f"Error downloading image: {str(e)}"

def clean_math_response(response):
    """Clean and format mathematical response while preserving structure"""
    # Remove LaTeX formatting but preserve structure
    response = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', response)
    response = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', response)
    response = re.sub(r'\$([^$]*)\$', r'\1', response)
    response = re.sub(r'\\boxed\{([^}]+)\}', r'**\1**', response)
    
    # Convert LaTeX symbols to Unicode
    symbol_replacements = {
        '\\theta': 'θ', '\\pi': 'π', '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ',
        '\\delta': 'δ', '\\omega': 'ω', '\\mu': 'μ', '\\sigma': 'σ', '\\Delta': 'Δ',
        '\\Sigma': 'Σ', '\\infty': '∞', '\\pm': '±', '\\times': '×', '\\div': '÷',
        '\\leq': '≤', '\\geq': '≥', '\\neq': '≠', '\\approx': '≈', '\\cdot': '·'
    }
    
    for latex_symbol, unicode_symbol in symbol_replacements.items():
        response = response.replace(latex_symbol, unicode_symbol)
    
    # Remove remaining LaTeX commands but preserve structure
    response = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', response)
    response = re.sub(r'\\[a-zA-Z]+', '', response)
    response = re.sub(r'\\', '', response)
    
    # Improve spacing around mathematical expressions
    response = re.sub(r'([a-zA-Z])\s*=\s*', r'\1 = ', response)
    response = re.sub(r'=\s*([a-zA-Z0-9])', r'= \1', response)
    
    # Clean up multiple line breaks but preserve intentional spacing
    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
    
    # Ensure proper spacing after headers and bullet points
    response = re.sub(r'(#+\s*[^\n]+)\n([^\n])', r'\1\n\n\2', response)
    response = re.sub(r'(\*\*[^*]+\*\*)\s*\n([^\n*])', r'\1\n\n\2', response)
    
    return response.strip()

def main():
    st.set_page_config(
        page_title="Multi-Model Math Solver", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧮 Multi-Model Math Solver")
    st.markdown("Choose from different AI models to solve your math problems!")
    
    # Initialize session state
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = {}
    
    # Sidebar for model selection and configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Model selection with availability check
        available_models = []
        
        # Always available models
        available_models.extend([
            "Llama Vision (3.2-90B) - Multimodal",
            "Llama 3.3-70B - Text Only"
        ])
        
        # Conditionally available models
        if LLAMA4_AVAILABLE:
            available_models.append("Llama 4 Scout (17B) - Multimodal")
        if QWEN2VL_AVAILABLE:
            available_models.append("Qwen2VL (72B) - Multimodal")
        
        model_choice = st.selectbox(
            "Choose Model:",
            available_models
        )
        
        model_key = {
            "Llama Vision (3.2-90B) - Multimodal": "llama_vision",
            "Llama 3.3-70B - Text Only": "llama33",
            "Llama 4 Scout (17B) - Multimodal": "llama4_scout",
            "Qwen2VL (72B) - Multimodal": "qwen2vl"
        }[model_choice]
        
        # Check if model supports images
        multimodal_models = ["llama_vision", "llama4_scout", "qwen2vl"]
        is_multimodal = model_key in multimodal_models
        
        st.info(f"📊 Model Type: {'Multimodal (Text + Image)' if is_multimodal else 'Text Only'}")
        
        # Model loading section
        st.subheader("Load Model")
        
        if model_key not in st.session_state.models_loaded:
            if model_key == "llama_vision":
                base_path = st.text_input("Base Model Path:", "meta-llama/Llama-3.2-90B-Vision-Instruct")
                st.info("📁 LoRA adapters will be automatically loaded if available")
                
                if st.button("🚀 Load Llama Vision", type="primary"):
                    with st.spinner("Loading Llama Vision model..."):
                        success = st.session_state.model_manager.load_llama_vision(base_path)
                        if success:
                            st.session_state.models_loaded[model_key] = True
                            st.balloons()
                        
            elif model_key == "llama33":
                base_path = st.text_input("Base Model Path:", "meta-llama/Llama-3.3-70B-Instruct")
                st.info("📁 LoRA adapters will be automatically loaded if available")
                
                if st.button("🚀 Load Llama 3.3", type="primary"):
                    with st.spinner("Loading Llama 3.3 model..."):
                        success = st.session_state.model_manager.load_llama33(base_path)
                        if success:
                            st.session_state.models_loaded[model_key] = True
                            st.balloons()
                            
            elif model_key == "llama4_scout":
                if st.button("🚀 Load Llama 4 Scout", type="primary"):
                    with st.spinner("Loading Llama 4 Scout model..."):
                        success = st.session_state.model_manager.load_llama4_scout()
                        if success:
                            st.session_state.models_loaded[model_key] = True
                            st.balloons()
                            
            elif model_key == "qwen2vl":
                if st.button("🚀 Load Qwen2VL", type="primary"):
                    with st.spinner("Loading Qwen2VL model..."):
                        success = st.session_state.model_manager.load_qwen2vl()
                        if success:
                            st.session_state.models_loaded[model_key] = True
                            st.balloons()
        else:
            st.success(f"✅ {model_choice.split(' -')[0]} is loaded!")
            if st.button("🔄 Reload Model"):
                del st.session_state.models_loaded[model_key]
                st.rerun()
        
        # Generation parameters
        st.subheader("⚙️ Generation Parameters")
        
        # Set default max_tokens based on model for better explanations
        default_max_tokens = 1024 if model_key in ["llama_vision", "llama33"] else 512
        max_tokens = st.slider("Max Tokens:", 256, 2048, default_max_tokens)
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
        
        # Image generation options for Llama 4 Scout
        enable_image_generation = False
        openai_api_key = None
        request_diagram = False
        
        if model_key == "llama4_scout":
            st.subheader("🎨 Image Generation (DALL-E)")
            if OPENAI_AVAILABLE:
                enable_image_generation = st.checkbox("Enable Image Generation", value=False)
                
                if enable_image_generation:
                    openai_api_key = st.text_input(
                        "OpenAI API Key:", 
                        type="password",
                        help="Enter your OpenAI API key to generate images with DALL-E 3"
                    )
                    request_diagram = st.checkbox("Always request diagram", value=False)
                    
                    if not openai_api_key:
                        st.warning("⚠️ Please enter your OpenAI API key to enable image generation")
            else:
                st.warning("⚠️ OpenAI library not available. Install with: pip install openai")
        
        # Model info
        st.subheader("📋 Current Model Info")
        if model_key in st.session_state.models_loaded:
            st.success("🟢 Model Loaded")
            if model_key in ["llama_vision", "llama33"]:
                lora_path = st.session_state.model_manager.lora_paths.get(model_key)
                if lora_path and os.path.exists(lora_path):
                    st.info("🔧 LoRA: Enabled")
                else:
                    st.info("🔧 LoRA: Not Available")
            
            # Output format info
            if model_key == "llama33":
                st.info("📋 Output: Structured with clear sections")
            elif model_key == "llama_vision":
                st.info("📋 Output: Detailed step-by-step explanations")
            else:
                st.info("📋 Output: Standard format")
        else:
            st.warning("🔴 Model Not Loaded")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💭 Ask Your Question")
        
        # Text input
        question = st.text_area(
            "Enter your math question:",
            height=150,
            placeholder="e.g., A simple pendulum with bob of mass m and conducting wire of length L swings under gravity through an angle 2θ. Calculate the maximum induced emf across the pendulum?"
        )
        
        # Image upload for multimodal models
        uploaded_image = None
        temp_image_path = None
        
        if is_multimodal:
            st.subheader("📸 Upload Image (Optional)")
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image containing diagrams, figures, or visual information related to your question."
            )
            
            if uploaded_image is not None:
                # Save uploaded image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(uploaded_image.getvalue())
                    temp_image_path = tmp_file.name
                
                # Display the image
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Generate button
        if st.button("🚀 Generate Solution", type="primary", use_container_width=True):
            if not question.strip():
                st.error("❌ Please enter a question!")
            elif model_key not in st.session_state.models_loaded:
                st.error(f"❌ Please load the {model_choice.split(' -')[0]} model first!")
            else:
                with st.spinner("🧠 Generating solution..."):
                    try:
                        response = None
                        generated_image = None
                        image_error = None
                        
                        # Call appropriate inference method
                        if model_key == "llama_vision":
                            response = st.session_state.model_manager.inference_llama_vision(
                                question, temp_image_path, max_tokens, temperature
                            )
                        elif model_key == "llama33":
                            response = st.session_state.model_manager.inference_llama33(
                                question, max_tokens, temperature
                            )
                        elif model_key == "llama4_scout":
                            response = st.session_state.model_manager.inference_llama4_scout(
                                question, temp_image_path, max_tokens, temperature, request_diagram
                            )
                            
                            # Handle image generation for Llama 4 Scout
                            if enable_image_generation and openai_api_key:
                                with st.spinner("🎨 Checking for image generation request..."):
                                    image_prompt = st.session_state.model_manager.extract_image_prompt(response)
                                    
                                    if image_prompt:
                                        st.info(f"🎨 Generating diagram with DALL-E...")
                                        st.info(f"**Image Prompt:** {image_prompt[:100]}...")
                                        
                                        with st.spinner("🎨 Generating image with DALL-E 3..."):
                                            image_url, error = st.session_state.model_manager.generate_image_with_dalle(
                                                image_prompt, openai_api_key
                                            )
                                            
                                            if image_url and not error:
                                                # Download and convert image
                                                generated_image, download_error = st.session_state.model_manager.download_and_convert_image(image_url)
                                                if download_error:
                                                    image_error = download_error
                                                else:
                                                    st.success("✅ Image generated successfully!")
                                            else:
                                                image_error = error
                                    else:
                                        st.info("ℹ️ No image generation requested by the model")
                                        
                        elif model_key == "qwen2vl":
                            response = st.session_state.model_manager.inference_qwen2vl(
                                question, temp_image_path, max_tokens, temperature
                            )
                        
                        # Clean and display response
                        if response:
                            cleaned_response = clean_math_response(response)
                            
                            st.subheader("🎯 Solution")
                            st.markdown(cleaned_response)
                            
                            # Display generated image if available
                            if generated_image:
                                st.subheader("🖼️ Generated Diagram")
                                st.image(generated_image, caption="AI-Generated Mathematical Diagram", use_column_width=True)
                            
                            # Display image generation error if any
                            if image_error:
                                st.error(f"🖼️ Image generation failed: {image_error}")
                            
                            # Create download data
                            download_content = cleaned_response
                            if generated_image:
                                download_content += "\n\n[Note: A mathematical diagram was generated to accompany this solution]"
                            
                            # Option to download solution
                            st.download_button(
                                label="📥 Download Solution",
                                data=download_content,
                                file_name=f"math_solution_{int(time.time())}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                            
                            # Option to download image if generated
                            if generated_image:
                                # Convert PIL image to bytes for download
                                img_buffer = BytesIO()
                                generated_image.save(img_buffer, format='PNG')
                                img_bytes = img_buffer.getvalue()
                                
                                st.download_button(
                                    label="🖼️ Download Diagram",
                                    data=img_bytes,
                                    file_name=f"math_diagram_{int(time.time())}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating solution: {str(e)}")
                        st.error("Please check your model paths and ensure all dependencies are installed correctly.")
                    
                    finally:
                        # Clean up temporary image file
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.unlink(temp_image_path)
    
    with col2:
        st.header("ℹ️ Model Information")
        
        model_info = {
            "llama_vision": {
                "name": "Llama Vision 3.2-90B",
                "type": "Multimodal (Vision + Text)",
                "strengths": ["Image understanding", "Mathematical diagrams", "Large parameter count"],
                "best_for": "Complex visual math problems",
                "fine_tuned": True,
                "image_generation": False
            },
            "llama33": {
                "name": "Llama 3.3-70B",
                "type": "Text Only",
                "strengths": ["Fast inference", "Strong reasoning", "Efficient"],
                "best_for": "Text-based math problems",
                "fine_tuned": True,
                "image_generation": False
            },
            "llama4_scout": {
                "name": "Llama 4 Scout 17B",
                "type": "Multimodal (Vision + Text)",
                "strengths": ["Latest architecture", "Balanced size", "Good image understanding", "DALL-E integration"],
                "best_for": "General math problems with images + diagram generation",
                "fine_tuned": False,
                "image_generation": True
            },
            "qwen2vl": {
                "name": "Qwen2VL 72B",
                "type": "Multimodal (Vision + Text)",
                "strengths": ["High accuracy", "Excellent vision", "Strong reasoning"],
                "best_for": "Complex visual reasoning",
                "fine_tuned": False,
                "image_generation": False
            }
        }
        
        info = model_info[model_key]
        
        st.markdown(f"**🤖 {info['name']}**")
        st.markdown(f"**📊 Type:** {info['type']}")
        
        if info['fine_tuned']:
            st.markdown("**🔧 Fine-tuned:** ✅ Yes")
        else:
            st.markdown("**🔧 Fine-tuned:** ❌ No")
            
        if info.get('image_generation', False):
            st.markdown("**🎨 Image Generation:** ✅ DALL-E 3")
        else:
            st.markdown("**🎨 Image Generation:** ❌ No")
            
        st.markdown("**💪 Strengths:**")
        for strength in info['strengths']:
            st.markdown(f"• {strength}")
        st.markdown(f"**🎯 Best for:** {info['best_for']}")
        
        # Usage tips
        st.subheader("💡 Usage Tips")
        
        # Model-specific tips
        if model_key == "llama33":
            st.markdown("""
            **📋 Enhanced Structured Output:**
            • Gets clear section headers (Problem Understanding, Solution Approach, etc.)
            • Step-by-step calculations with explanations
            • Organized final answer with units
            • Best for detailed text-based problems
            """)
        elif model_key == "llama_vision":
            st.markdown("""
            **📋 Detailed Visual Analysis:**
            • Thorough image analysis when provided
            • Structured problem breakdown
            • Clear step-by-step solutions
            • Best for problems with diagrams/figures
            """)
        
        if is_multimodal:
            st.markdown("""
            **📷 General Multimodal Tips:**
            • Upload clear images with readable text and diagrams
            • Describe what you see in the image within your question
            • Use specific mathematical terminology
            • For geometry problems, include measurements from the image
            """)
            
            if model_key == "llama4_scout":
                st.markdown("""
                **🎨 Image Generation Tips:**
                • Enable image generation for visual solutions
                • The model will automatically decide when to create diagrams
                • Use "Always request diagram" for guaranteed image output
                • Provide your OpenAI API key for DALL-E 3 access
                """)
        else:
            st.markdown("""
            **📝 Text-Only Tips:**
            • Be specific and detailed in your questions
            • Include all given information clearly
            • State what you need to find explicitly
            • Use proper mathematical notation in text
            """)
        
        # System info
        st.subheader("⚙️ System Status")
        st.markdown(f"**🐍 Python:** {sys.version.split()[0]}")
        st.markdown(f"**🔥 PyTorch:** {torch.__version__}")
        st.markdown(f"**🤗 Transformers:** Available")
        st.markdown(f"**🦙 Llama4:** {'✅' if LLAMA4_AVAILABLE else '❌'}")
        st.markdown(f"**🔍 Qwen2VL:** {'✅' if QWEN2VL_AVAILABLE else '❌'}")
        st.markdown(f"**🎨 OpenAI:** {'✅' if OPENAI_AVAILABLE else '❌'}")

if __name__ == "__main__":
    main()