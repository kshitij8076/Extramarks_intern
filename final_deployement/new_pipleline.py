import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText, pipeline
from PIL import Image
import base64
import io
import re
import requests
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionSolvingPipeline:
    def __init__(self):
        """Initialize the question solving pipeline with all models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        self.processors = {}
        
        # Load base reasoning model
        self.load_base_model()
        
        # Model configurations for different specializations
        self.model_configs = {
            "math_physics": "Qwen/Qwen2.5-Math-7B-Instruct",
            "biology_chemistry": "google/medgemma-4b-it",  # Updated to use MedGemma
            "prover": "deepseek-ai/deepseek-math-7b-rl",
            # "vision": "Qwen/Qwen2-VL-72B-Instruct",
            "vision": "Qwen/QVQ-72B-Preview",
            "general": "meta-llama/Llama-3.1-8B-Instruct"
        }
        
        # Difficulty levels
        self.difficulty_levels = {
            1: "too_easy",
            2: "easy", 
            3: "medium_high_school",
            4: "hard_college",
            5: "competitive_jee"
        }
        
        # Subject categories
        self.subjects = ["general", "maths", "physics", "biology", "chemistry", "prover", "vision"]
        
    def load_base_model(self):
        """Load the base Phi-4 reasoning model."""
        try:
            model_name = "microsoft/Phi-4"
            self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
    
    def load_specialized_model(self, model_type: str):
        """Load specialized models on demand."""
        if model_type in self.models:
            return self.models[model_type], self.tokenizers.get(model_type), self.processors.get(model_type)
        
        try:
            model_name = self.model_configs[model_type]
            
            # Special handling for MedGemma (multimodal model)
            if model_type == "biology_chemistry" and "medgemma" in model_name:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(model_name)
                
                self.models[model_type] = model
                self.processors[model_type] = processor
                
                logger.info(f"Loaded {model_type} model (MedGemma): {model_name}")
                return model, None, processor
            else:
                # Standard text-only models
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                self.models[model_type] = model
                self.tokenizers[model_type] = tokenizer
                
                logger.info(f"Loaded {model_type} model: {model_name}")
                return model, tokenizer, None
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {e}")
            return None, None, None
    
    def has_image(self, question_data: Dict) -> bool:
        """Check if the question contains an image."""
        return "image" in question_data and question_data["image"] is not None
    
    def classify_difficulty_and_subject(self, question: str) -> Tuple[int, str]:
        """Classify question difficulty (1-5) and subject using base model."""
        
        classification_prompt = f"""
You are an expert question classifier for academic questions. Analyze the following question and provide:
1. Difficulty level (1-5):
   - Level 1: Too easy (basic arithmetic, simple facts)
   - Level 2: Easy (elementary to middle school level)
   - Level 3: Medium (high school level, NCERT standard)
   - Level 4: Hard (JEE Mains level, competitive exam standard)
   - Level 5: Competitive (JEE Advanced level, highly complex)

2. Subject category:
   - general: General knowledge, everyday questions
   - maths: Mathematics problems 
   - physics: Physics problems
   - biology: Biology questions
   - chemistry: Chemistry questions
   - prover: Mathematical proofs, theorem proving

Classification Rules:
1. **prover**: If the question asks to "prove", "show that", "demonstrate", "verify", "establish", or "derive" something mathematical → classify as "prover"
2. **maths**: If it involves mathematical calculations, equations, algebra, calculus, geometry, trigonometry, statistics, probability, derivatives, integrals, matrices, vectors → classify as "maths"
3. **physics**: If it involves physics concepts like force, energy, motion, velocity, acceleration, electricity, magnetism, thermodynamics, optics, waves, quantum mechanics, relativity, circuits, electromagnetic fields → classify as "physics"
4. **biology**: If it involves living organisms, cells, DNA, genetics, anatomy, physiology, ecology, evolution, photosynthesis, respiration, reproduction, classification of organisms, plant/animal structures → classify as "biology"
5. **chemistry**: If it involves chemical reactions, periodic table, elements, compounds, acids, bases, pH, molecular structure, chemical bonding, stoichiometry, organic chemistry, inorganic chemistry, physical chemistry → classify as "chemistry"
6. **general**: If it involves general knowledge, current affairs, history, geography, literature, everyday life questions, or doesn't fit into the above scientific categories → classify as "general"

Question: {question}

Respond in this exact format:
Difficulty: [1-5]
Subject: [subject_name]

Examples for reference:

Level 1 example:
Question: "What is 2+2?"
Difficulty: 1
Subject: maths

Level 2 example:
Question: "Find the area of a rectangle with length 5 cm and width 3 cm"
Difficulty: 2
Subject: maths

Level 3 example:
Question: "Solve the quadratic equation x² + 5x + 6 = 0"
Difficulty: 3
Subject: maths

Level 4 example (JEE Mains level):
Question: "Bag B1 contains 6 white and 4 blue balls, Bag B2 contains 4 white and 6 blue balls, and Bag B3 contains 5 white and 5 blue balls. One of the bags is selected at random and a ball is drawn from it. If the ball is white, then the probability that the ball is drawn from Bag B2 is:"
Difficulty: 4
Subject: maths

Level 5 example (JEE Advanced level):
Question: "An electron in a hydrogen atom undergoes a transition from an orbit with quantum number ni to another with quantum number nf. Vi and Vf are respectively the initial and final potential energies of the electron. If Vi/Vf = 6.25, then the smallest possible nf is"
Difficulty: 5
Subject: physics

Additional subject examples:
Question: "What is the pH of a 0.1 M HCl solution?"
Difficulty: 3
Subject: chemistry

Question: "Explain the process of mitosis in plant cells"
Difficulty: 3
Subject: biology

Question: "Prove that the sum of angles in a triangle is 180 degrees"
Difficulty: 4
Subject: prover

Question: "Who was the first Prime Minister of India?"
Difficulty: 1
Subject: general

Now classify the given question:
"""
        
        try:
            inputs = self.base_tokenizer(classification_prompt, return_tensors="pt", truncation=True, max_length=1500)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.base_tokenizer.eos_token_id
                )
            
            response = self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(classification_prompt, "").strip()
            
            # Parse response
            difficulty_match = re.search(r"Difficulty:\s*(\d+)", response)
            subject_match = re.search(r"Subject:\s*(\w+)", response)
            
            difficulty = int(difficulty_match.group(1)) if difficulty_match else 3
            subject = subject_match.group(1).lower() if subject_match else "general"
            
            # Validate ranges
            difficulty = max(1, min(5, difficulty))
            if subject not in self.subjects:
                subject = "general"
            
            logger.info(f"Classified - Difficulty: {difficulty}, Subject: {subject}")
            return difficulty, subject
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return 3, "general"
    
    def solve_with_base_model(self, question: str) -> str:
        """Solve easy questions (Level 1-2) with base model."""
        solve_prompt = f"""<|system|>
You are a helpful AI assistant. Solve the given question step by step and provide only the solution for the given question.

<|user|>
{question}

<|assistant|>
I'll solve this step by step:

"""
        
        try:
            inputs = self.base_tokenizer(solve_prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.base_tokenizer.eos_token_id
                )
            
            response = self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = response.replace(solve_prompt, "").strip()
            
            # Clean up response
            stop_patterns = ["<|user|>", "<|system|>", "Question:", "I'll solve this step by step:"]
            for pattern in stop_patterns:
                if pattern in solution:
                    solution = solution.split(pattern)[0].strip()
            
            return solution
            
        except Exception as e:
            logger.error(f"Error solving with base model: {e}")
            return "Sorry, I couldn't solve this question."
    
    def solve_with_medgemma(self, question: str) -> str:
        """Solve biology/chemistry questions with MedGemma-4B-IT."""
        model, tokenizer, processor = self.load_specialized_model("biology_chemistry")
        
        if model is None or processor is None:
            return "Sorry, the MedGemma model is not available."
        
        try:
            # Create messages for MedGemma
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert in biology and chemistry. Provide detailed, accurate answers to scientific questions."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}]
                }
            ]
            
            # Process the input
            inputs = processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response
            with torch.inference_mode():
                generation = model.generate(
                    **inputs, 
                    max_new_tokens=1024, 
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9
                )
                generation = generation[0][input_len:]
            
            # Decode the response
            solution = processor.decode(generation, skip_special_tokens=True)
            return solution.strip()
            
        except Exception as e:
            logger.error(f"Error solving with MedGemma: {e}")
            return "Sorry, I couldn't solve this question with the MedGemma model."
    
    def solve_with_specialized_model(self, question: str, model_type: str) -> str:
        """Solve questions with specialized models."""
        # Special handling for MedGemma
        if model_type == "biology_chemistry":
            return self.solve_with_medgemma(question)
        
        model, tokenizer, processor = self.load_specialized_model(model_type)
        
        if model is None or tokenizer is None:
            return "Sorry, the specialized model is not available."
        
        # Create appropriate prompt based on model type
        if model_type == "math_physics":
            prompt = f"Solve this mathematics/physics problem step by step:\n\n{question}\n\nSolution:"
        elif model_type == "prover":
            prompt = f"Provide a mathematical proof for:\n\n{question}\n\nProof:"
        else:  # general
            prompt = f"Answer this question comprehensively:\n\n{question}\n\nAnswer:"
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = response.replace(prompt, "").strip()
            
            return solution
            
        except Exception as e:
            logger.error(f"Error solving with {model_type} model: {e}")
            return "Sorry, I couldn't solve this question with the specialized model."
    
    def solve_vision_question(self, question: str, image_data: Optional[str] = None) -> str:
        """Solve questions with images using vision model."""
        try:
            # For now, return a placeholder response
            return f"Vision-based solution for: {question}\n(Vision model integration needed)"
            
        except Exception as e:
            logger.error(f"Error with vision model: {e}")
            return "Sorry, I couldn't process the image-based question."
    
    def solve_question(self, question_data: Dict) -> Dict:
        """Main pipeline function to solve questions."""
        question = question_data.get("question", "")
        
        if not question:
            return {"error": "No question provided"}
        
        # Check for images first
        if self.has_image(question_data):
            logger.info("Processing question with image using vision model")
            solution = self.solve_vision_question(question, question_data.get("image"))
            return {
                "question": question,
                "solution": solution,
                "model_used": "vision",
                "difficulty": "N/A",
                "subject": "vision"
            }
        
        # Classify difficulty and subject
        difficulty, subject = self.classify_difficulty_and_subject(question)
        
        # Route to appropriate model based on classification
        if difficulty <= 2:
            # Easy questions - use base model
            logger.info(f"Solving Level {difficulty} question with base model")
            solution = self.solve_with_base_model(question)
            model_used = "base"
            
        elif difficulty >= 3:
            # Harder questions - use specialized models
            if subject in ["maths", "physics"]:
                logger.info(f"Solving Level {difficulty} {subject} question with math/physics model")
                solution = self.solve_with_specialized_model(question, "math_physics")
                model_used = "math_physics"
                
            elif subject in ["biology", "chemistry"]:
                logger.info(f"Solving Level {difficulty} {subject} question with MedGemma model")
                solution = self.solve_with_specialized_model(question, "biology_chemistry")
                model_used = "biology_chemistry (MedGemma)"
                
            elif subject == "prover":
                logger.info(f"Solving Level {difficulty} proof question with prover model")
                solution = self.solve_with_specialized_model(question, "prover")
                model_used = "prover"
                
            else:  # general or unknown
                logger.info(f"Solving Level {difficulty} general question with general model")
                solution = self.solve_with_specialized_model(question, "general")
                model_used = "general"
        
        return {
            "question": question,
            "solution": solution,
            "difficulty": difficulty,
            "subject": subject,
            "model_used": model_used
        }

# Usage example and API wrapper
class QuestionSolvingAPI:
    def __init__(self):
        self.pipeline = QuestionSolvingPipeline()
    
    def process_question(self, question: str, image: Optional[str] = None) -> Dict:
        """
        Process a question through the pipeline.
        
        Args:
            question: The question text
            image: Base64 encoded image (optional)
        
        Returns:
            Dictionary with solution and metadata
        """
        question_data = {
            "question": question,
            "image": image
        }
        
        return self.pipeline.solve_question(question_data)

# Example usage
if __name__ == "__main__":
    # Initialize the API
    api = QuestionSolvingAPI()
    
    # Example questions
    test_questions = [
        # 'An injury sustained by the hypothalamus is most likely to interrupt',
        'What is the pH of a 0.1 M HCl` solution?',
        # 'Explain the process of photosynthesis in detail',
        'Three defective oranges are accidently mixed with seven good ones and on looking at them, it is not possible to differentiate between them. Two oranges are drawn at random from the lot. If x denote the number of defective oranges, then the variance of x is',
        'A researcher observes that the rate of an enzyme-catalyzed reaction is significantly reduced when the concentration of the substrate is high. Which of the following factors is most likely responsible for this observation? Options:A) Non-competitive inhibition B) Competitive inhibition C) Allosteric regulation D) Feedback inhibition ',
        'A solution is made by mixing one mole of volatile liquid A  with 3 moles of volatile liquid B. The vapour pressure of pure A is 200 mm Hg and that of the solution is 500 mm Hg . The vapour pressure of pure B and the least volatile component of the solution, respectively, are:'
    ]

    # image_questions = [
    #     'From a point Q, the length of the tangent to a circle is 24 cm and the distance of Q from the centre is 25 cm. Find the radius of the circle.'
    # ]
    
    # image_pth = '/home/karang/dikshant/extramarks_239/q1.png'

    # result = api.process_question(image_questions[0],image_pth)
    # print(f"Difficulty: Level {result['difficulty']}")
    # print(f"Subject: {result['subject']}")
    # print(f"Model Used: {result['model_used']}")
    # print(f"Solution: {result['solution']}")
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"{'='*50}")
        
        result = api.process_question(question)
        
        print(f"Difficulty: Level {result['difficulty']}")
        print(f"Subject: {result['subject']}")
        print(f"Model Used: {result['model_used']}")
        print(f"Solution: {result['solution']}")
