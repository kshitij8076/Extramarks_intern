import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
    AutoModelForVision2Seq, LlavaForConditionalGeneration
)
from PIL import Image
import requests
import re
import logging
from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCategory(Enum):
    PROVER = "prover"
    MATH = "math"
    PHYSICS = "physics"
    BIOLOGY = "biology"
    VISION = "vision"
    TEXT_LLM = "text_llm"

@dataclass
class ModelConfig:
    name: str
    model_id: str
    category: ModelCategory
    supports_vision: bool = False
    priority: int = 1

class Phi4QuestionClassifier:
    """Question classifier using Microsoft Phi-4-reasoning as base model"""
    
    def __init__(self, model_name="microsoft/Phi-4"):
        """Initialize with Phi-4-reasoning (fallback to Phi-4 if not available)"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
       
        model_options = [
            "microsoft/Phi-4-reasoning",  
        ]
        
        self.model = None
        self.tokenizer = None
        
        for model_name in model_options:
            try:
                logger.info(f"Attempting to load {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                logger.info(f"Successfully loaded {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if self.model is None:
            raise RuntimeError("Failed to load any Phi model variant")
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.categories = ["math", "physics", "biology", "science", "prover"]
    
    def create_classification_prompt(self, question: str) -> str:
        """Create a structured prompt for Phi-4-reasoning classification"""
        prompt = f"""<|system|>
You are an expert question classifier for educational content. Your task is to classify questions into exactly one of these categories:

- **math**: Pure mathematics problems (algebra, calculus, geometry, statistics, equations, derivatives, integrals)
- **physics**: Physics problems (mechanics, thermodynamics, electromagnetism, optics, quantum physics, forces, motion, energy)
- **biology**: Life sciences questions (cells, DNA, organisms, anatomy, physiology, ecology, evolution)
- **science**: General science or chemistry (chemical reactions, periodic table, acids/bases, compounds)
- **prover**: Mathematical proof requests (prove, demonstrate, show that, verify mathematically)

Classification Rules:
1. If the question asks to "prove", "show that", "demonstrate", or "verify" something mathematical → classify as "prover"
2. If it involves mathematical calculations, equations, or mathematical concepts → classify as "math"
3. If it involves physics concepts like force, energy, motion, electricity → classify as "physics"
4. If it involves living organisms, cells, biological processes → classify as "biology"
5. If it involves chemistry or general science concepts → classify as "science"

Respond with ONLY the category name (math, physics, biology, science, or prover).

<|user|>
Classify this question: "{question}"

<|assistant|>
"""
        return prompt
    
    def classify_question(self, question: str, max_length: int = 1024, temperature: float = 0.1) -> str:
        """Classify a question using Phi-4-reasoning"""
        prompt = self.create_classification_prompt(question)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("<|assistant|>")[-1].strip().lower()
        
        # Clean and validate response
        response = re.sub(r'[^\w\s]', '', response).strip()
        
        # Find the category in the response
        for category in self.categories:
            if category in response:
                return category
        
        # Fallback classification
        return self.fallback_classification(question)
    
    def fallback_classification(self, question: str) -> str:
        """Fallback classification using keyword matching"""
        question_lower = question.lower()
        
        # Proof keywords - highest priority
        proof_keywords = ['prove', 'proof', 'demonstrate', 'show that', 'verify', 'establish']
        if any(keyword in question_lower for keyword in proof_keywords):
            return 'prover'
        
        # Math keywords
        math_keywords = ['calculate', 'solve', 'equation', 'algebra', 'geometry', 'trigonometry',
                        'calculus', 'derivative', 'integral', 'matrix', 'probability', 'statistics',
                        'sum', 'product', 'ratio', 'percentage', 'fraction', 'formula']
        
        # Physics keywords
        physics_keywords = ['force', 'velocity', 'acceleration', 'momentum', 'energy', 'power',
                           'electric', 'magnetic', 'quantum', 'wave', 'frequency', 'mass',
                           'gravity', 'pressure', 'temperature', 'heat', 'light', 'motion',
                           'newton', 'joule', 'watt', 'voltage', 'current', 'resistance']
        
        # Biology keywords
        biology_keywords = ['cell', 'dna', 'gene', 'organism', 'evolution', 'ecology',
                           'anatomy', 'physiology', 'bacteria', 'virus', 'protein',
                           'enzyme', 'metabolism', 'reproduction', 'species', 'photosynthesis',
                           'respiration', 'mitosis', 'meiosis', 'chromosome']
        
        # Count keyword matches
        math_score = sum(1 for keyword in math_keywords if keyword in question_lower)
        physics_score = sum(1 for keyword in physics_keywords if keyword in question_lower)
        biology_score = sum(1 for keyword in biology_keywords if keyword in question_lower)
        
        # Return category with highest score
        scores = {'math': math_score, 'physics': physics_score, 'biology': biology_score}
        max_category = max(scores, key=scores.get)
        
        return max_category if scores[max_category] > 0 else 'science'

class CompletePhi4Pipeline:
    """Complete pipeline using Phi-4-reasoning for classification and specialized models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}
        
        # Initialize the Phi-4-reasoning classifier
        logger.info("Initializing Phi-4-reasoning classifier...")
        self.classifier = Phi4QuestionClassifier()
        
        # Define model configurations
        self.model_configs = self._initialize_model_configs()
        
        logger.info("Pipeline initialized successfully!")
    
    def _initialize_model_configs(self) -> Dict[str, List[ModelConfig]]:
        """Initialize all model configurations"""
        return {
            "prover": [
                ModelConfig("DeepSeek-Prover-V2", "deepseek-ai/DeepSeek-Prover-V2-7B", ModelCategory.PROVER, priority=1)
            ],
            "math": [
                ModelConfig("Qwen2.5-Math-7B", "Qwen/Qwen2.5-Math-7B-Instruct", ModelCategory.MATH, priority=1),
                ModelConfig("DeepSeek-Math-7B", "deepseek-ai/deepseek-math-7b-rl", ModelCategory.MATH, priority=2),
                ModelConfig("Qwen2.5-Math-72B", "Qwen/Qwen2.5-Math-72B-Instruct", ModelCategory.MATH, priority=3)
            ],
            "physics": [
                ModelConfig("Qwen2.5-Math-7B", "Qwen/Qwen2.5-Math-7B-Instruct", ModelCategory.PHYSICS, priority=1),
                ModelConfig("DeepSeek-Math-7B", "deepseek-ai/deepseek-math-7b-rl", ModelCategory.PHYSICS, priority=2),
                ModelConfig("Qwen2.5-Math-72B", "Qwen/Qwen2.5-Math-72B-Instruct", ModelCategory.PHYSICS, priority=3)
            ],
            "biology": [
                ModelConfig("MedGemma-4B", "google/medgemma-4b-it", ModelCategory.BIOLOGY, supports_vision=True, priority=1)
            ],
            "science": [
                ModelConfig("Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct", ModelCategory.TEXT_LLM, priority=1),
                ModelConfig("Phi-4", "microsoft/Phi-4", ModelCategory.TEXT_LLM, priority=2)
            ],
            "vision": [
                ModelConfig("QVQ-72B", "Qwen/QVQ-72B-Preview", ModelCategory.VISION, supports_vision=True, priority=1),
                ModelConfig("DeepSeek-VL2", "deepseek-ai/deepseek-vl2", ModelCategory.VISION, supports_vision=True, priority=2),
                ModelConfig("Qwen2.5-VL-32B", "Qwen/Qwen2.5-VL-32B-Instruct", ModelCategory.VISION, supports_vision=True, priority=3),
                ModelConfig("Qwen2.5-VL-72B", "Qwen/Qwen2.5-VL-72B-Instruct", ModelCategory.VISION, supports_vision=True, priority=4),
                ModelConfig("Llama-3.2-90B-Vision", "meta-llama/Llama-3.2-90B-Vision-Instruct", ModelCategory.VISION, supports_vision=True, priority=5)
            ],
            "text_llm": [
                ModelConfig("DeepSeek-R1-7B", "deepseek-ai/DeepSeek-R1-Zero", ModelCategory.TEXT_LLM, priority=1),
                ModelConfig("Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct", ModelCategory.TEXT_LLM, priority=2),
                ModelConfig("QwQ-32B", "Qwen/QwQ-32B-Preview", ModelCategory.TEXT_LLM, priority=3),
                ModelConfig("Phi-4", "microsoft/Phi-4", ModelCategory.TEXT_LLM, priority=4),
                ModelConfig("Llama-3.3-70B", "meta-llama/Llama-3.3-70B-Instruct", ModelCategory.TEXT_LLM, priority=5),
                ModelConfig("Llama-3.2-11B", "meta-llama/Llama-3.2-11B-Vision-Instruct", ModelCategory.TEXT_LLM, priority=6)
            ]
        }
    
    def _detect_image_input(self, image: Optional[Union[str, Image.Image]]) -> bool:
        """Detect if image input is provided"""
        return image is not None
    
    def _get_best_model(self, category: str, requires_vision: bool = False) -> ModelConfig:
        """Get the best model for a category and vision requirement"""
        
        if requires_vision:
            # For image inputs, prioritize vision models or biology model with vision support
            if category == "biology":
                # Biology questions with images go to MedGemma (supports vision)
                return self.model_configs["biology"][0]
            else:
                # Other categories with images go to vision models
                candidates = self.model_configs["vision"]
        else:
            # Text-only questions go to category-specific models
            candidates = self.model_configs.get(category, self.model_configs["text_llm"])
        
        # Sort by priority and return best
        candidates.sort(key=lambda x: x.priority)
        return candidates[0] if candidates else self.model_configs["text_llm"][0]
    
    def _load_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Load a specific model configuration"""
        if config.model_id in self.loaded_models:
            return self.loaded_models[config.model_id]
        
        logger.info(f"Loading {config.name} ({config.model_id})...")
        
        try:
            # Load tokenizer/processor
            if config.supports_vision:
                try:
                    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
                    tokenizer = None
                except:
                    # Fallback for models without processor
                    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
                    processor = None
            else:
                tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
                processor = None
            
            # Load model
            if config.supports_vision and processor is not None:
                if "llava" in config.model_id.lower():
                    model = LlavaForConditionalGeneration.from_pretrained(
                        config.model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                else:
                    try:
                        model = AutoModelForVision2Seq.from_pretrained(
                            config.model_id,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                    except:
                        # Fallback for vision models
                        model = AutoModelForCausalLM.from_pretrained(
                            config.model_id,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # Set pad token if needed
            if tokenizer and tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_dict = {
                "model": model,
                "tokenizer": tokenizer,
                "processor": processor,
                "config": config
            }
            
            self.loaded_models[config.model_id] = model_dict
            logger.info(f"Successfully loaded {config.name}")
            return model_dict
            
        except Exception as e:
            logger.error(f"Failed to load {config.name}: {e}")
            # Return fallback model
            return self._load_fallback_model()
    
    def _load_fallback_model(self) -> Dict[str, Any]:
        """Load a simple fallback model"""
        logger.info("Loading fallback model...")
        try:
            fallback_id = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(fallback_id)
            model = AutoModelForCausalLM.from_pretrained(fallback_id)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            config = ModelConfig("Fallback", fallback_id, ModelCategory.TEXT_LLM)
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "processor": None,
                "config": config
            }
        except Exception as e:
            logger.error(f"Even fallback failed: {e}")
            raise e
    
    def _create_specialized_prompt(self, question: str, category: str, config: ModelConfig) -> str:
        """Create specialized prompts for different model types"""
        
        if config.category == ModelCategory.PROVER:
            return f"""You are a mathematical proof expert. Provide a rigorous, step-by-step proof for:

{question}

Structure your proof with:
1. Clear assumptions and definitions
2. Logical step-by-step reasoning
3. Mathematical justification for each step
4. Clear conclusion

Proof:"""

        elif config.category == ModelCategory.MATH:
            return f"""You are a mathematics expert. Solve this problem step by step with clear explanations:

{question}

Solution:"""

        elif config.category == ModelCategory.PHYSICS:
            return f"""You are a physics expert. Solve this physics problem with detailed explanations:

{question}

Include:
- Given information
- Relevant physics principles
- Step-by-step calculations
- Final answer with units

Solution:"""

        elif config.category == ModelCategory.BIOLOGY:
            return f"""You are a biology and medical expert. Answer this question with scientific accuracy:

{question}

Provide detailed biological explanations with relevant concepts and examples.

Answer:"""

        elif config.category == ModelCategory.VISION:
            return f"""Analyze the image and answer this question:

{question}

Provide detailed observations and explanations based on what you see in the image.

Answer:"""

        else:  # General text LLM
            return f"""Answer the following question accurately and helpfully:

{question}

Answer:"""
    
    def _generate_response(self, 
                          model_dict: Dict[str, Any], 
                          prompt: str, 
                          image: Optional[Union[str, Image.Image]] = None,
                          max_length: int = 1024,
                          temperature: float = 0.7) -> str:
        """Generate response using the loaded model"""
        
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        processor = model_dict["processor"]
        
        try:
            if image is not None and processor is not None:
                # Vision model processing
                if isinstance(image, str):
                    if image.startswith('http'):
                        image = Image.open(requests.get(image, stream=True).raw)
                    else:
                        image = Image.open(image)
                
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            else:
                # Text-only processing
                if tokenizer is None:
                    tokenizer = processor  # Some models use processor as tokenizer
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=tokenizer.eos_token_id if tokenizer else model.config.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id if tokenizer else model.config.eos_token_id
                )
            
            # Decode response
            if tokenizer:
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                full_response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            response = full_response[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while generating the response: {str(e)}"
    
    def ask(self, 
            question: str, 
            image: Optional[Union[str, Image.Image]] = None,
            max_length: int = 1024,
            temperature: float = 0.7) -> Dict[str, Any]:
        """
        Main function to ask questions through the pipeline
        
        Args:
            question: The question to ask
            image: Optional image input (path or PIL Image)
            max_length: Maximum response length
            temperature: Generation temperature
        
        Returns:
            Dictionary with response and metadata
        """
        
        logger.info(f"Processing question: {question[:50]}...")
        
        try:
            # Step 1: Check if image is provided
            has_image = self._detect_image_input(image)
            logger.info(f"Image provided: {has_image}")
            
            # Step 2: Classify the question using Phi-4-reasoning
            category = self.classifier.classify_question(question)
            logger.info(f"Question classified as: {category}")
            
            # Step 3: Select appropriate model
            model_config = self._get_best_model(category, has_image)
            logger.info(f"Selected model: {model_config.name}")
            
            # Step 4: Load the model
            model_dict = self._load_model(model_config)
            
            # Step 5: Create specialized prompt
            prompt = self._create_specialized_prompt(question, category, model_config)
            
            # Step 6: Generate response
            response = self._generate_response(
                model_dict, prompt, image, max_length, temperature
            )
            
            return {
                "success": True,
                "question": question,
                "category": category,
                "model_used": model_config.name,
                "model_id": model_config.model_id,
                "has_image": has_image,
                "response": response,
                "classifier_model": "microsoft/Phi-4-reasoning"
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "classifier_model": "microsoft/Phi-4-reasoning"
            }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline"""
        return {
            "classifier_model": "microsoft/Phi-4-reasoning",
            "available_categories": list(self.model_configs.keys()),
            "loaded_models": list(self.loaded_models.keys()),
            "model_configs": {
                category: [
                    {
                        "name": config.name,
                        "model_id": config.model_id,
                        "supports_vision": config.supports_vision,
                        "priority": config.priority
                    }
                    for config in configs
                ]
                for category, configs in self.model_configs.items()
            }
        }

# Convenience functions for easy usage
def create_pipeline() -> CompletePhi4Pipeline:
    """Create and return a new pipeline instance"""
    return CompletePhi4Pipeline()

def quick_ask(question: str, image: Optional[str] = None) -> str:
    """Quick function to ask a question"""
    global _global_pipeline
    
    if '_global_pipeline' not in globals():
        _global_pipeline = create_pipeline()
    
    result = _global_pipeline.ask(question, image)
    return result.get("response", f"Error: {result.get('error', 'Unknown error')}")

# Demo and testing functions
def demo():
    """Demonstration of the complete pipeline"""
    print("🤖 Initializing Complete Phi-4-Reasoning Pipeline...")
    print("=" * 60)
    
    pipeline = create_pipeline()
    
    # Test cases covering all categories
    test_cases = [
        {
            "question": "Prove that the square root of 2 is irrational",
            "expected_category": "prover",
            "image": None
        },
        {
            "question": "Solve the equation x^2 + 5x + 6 = 0",
            "expected_category": "math",
            "image": None
        },
        {
            "question": "Calculate the force needed to accelerate a 10kg object at 5 m/s²",
            "expected_category": "physics",
            "image": None
        },
        {
            "question": "Explain how photosynthesis works in plants",
            "expected_category": "biology",
            "image": None
        },
        {
            "question": "What is the chemical formula for water?",
            "expected_category": "science",
            "image": None
        }
    ]
    
    print("\n📋 Testing Pipeline with Sample Questions...")
    print("=" * 60)
    
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"Question: {test_case['question']}")
        print(f"Expected Category: {test_case['expected_category']}")
        print("-" * 40)
        
        result = pipeline.ask(test_case["question"], test_case["image"])
        
        if result["success"]:
            print(f"✅ Classification: {result['category']}")
            print(f"🤖 Model Used: {result['model_used']}")
            print(f"📝 Response Preview: {result['response'][:150]}...")
            
            # Check if classification matches expectation
            if result['category'] == test_case['expected_category']:
                print("🎯 Classification: CORRECT")
            else:
                print("⚠️  Classification: DIFFERENT THAN EXPECTED")
        else:
            print(f"❌ Error: {result['error']}")
        
        print()
    
    # Show pipeline info
    print("\n📊 Pipeline Information:")
    print("=" * 60)
    info = pipeline.get_pipeline_info()
    print(f"Classifier Model: {info['classifier_model']}")
    print(f"Available Categories: {', '.join(info['available_categories'])}")
    print(f"Currently Loaded Models: {len(info['loaded_models'])}")
    
    return pipeline

def interactive_mode():
    """Interactive mode for testing"""
    print("\n🎮 Interactive Mode")
    print("=" * 40)
    print("Commands:")
    print("  ask <question>     - Ask any question")
    print("  info              - Show pipeline info")
    print("  help              - Show this help")
    print("  quit              - Exit")
    print()
    
    pipeline = create_pipeline()
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() in ['help', 'h']:
                print("\nCommands:")
                print("  ask <question>     - Ask any question")
                print("  info              - Show pipeline info")
                print("  help              - Show this help")
                print("  quit              - Exit")
                continue
            
            if user_input.lower() == 'info':
                info = pipeline.get_pipeline_info()
                print(json.dumps(info, indent=2))
                continue
            
            if user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                if question:
                    print(f"\n🤔 Processing: {question}")
                    result = pipeline.ask(question)
                    
                    if result["success"]:
                        print(f"🏷️  Category: {result['category']}")
                        print(f"🤖 Model: {result['model_used']}")
                        print(f"💬 Response:\n{result['response']}")
                    else:
                        print(f"❌ Error: {result['error']}")
                else:
                    print("Please provide a question after 'ask'")
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        interactive_mode()
    else:
        # Run demo
        demo()
        
        # Offer interactive mode
        print(f"\n🎮 Want to try interactive mode?")
        response = input("Type 'y' for interactive mode, any other key to exit: ").strip().lower()
        if response == 'y':
            interactive_mode()
    # interactive_mode()