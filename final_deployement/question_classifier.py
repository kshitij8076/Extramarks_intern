import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json

class QuestionClassifier:
    def __init__(self, model_name="microsoft/Phi-4"):
        """
        Initialize the question classifier with Phi-4-reasoning model
        Note: Phi-4-reasoning might not be publicly available yet, using Phi-4 as fallback
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to microsoft/Phi-3-mini-4k-instruct")
            model_name = "microsoft/Phi-3-mini-4k-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.categories = ["math", "science", "biology", "physics"]
    
    def create_classification_prompt(self, question):
        """Create a prompt for question classification"""
        prompt = f"""<|system|>
You are an expert question classifier. Your task is to classify questions into exactly one of these categories: math, science, biology, physics.

Rules:
- Math: arithmetic, algebra, calculus, geometry, statistics, mathematical problems
- Physics: mechanics, thermodynamics, electromagnetism, quantum physics, optics, forces, motion
- Biology: life sciences, anatomy, genetics, ecology, cell biology, organisms
- Science: general science, chemistry, earth science, or questions that don't fit specifically into math, biology, or physics

Respond with ONLY the category name (math, science, biology, or physics).

<|user|>
Classify this question: "{question}"

<|assistant|>
"""
        return prompt
    
    def classify_question(self, question, max_length=512, temperature=0.1):
        """
        Classify a question into one of the predefined categories
        """
        prompt = self.create_classification_prompt(question)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,  # We only need a short classification response
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = full_response.split("<|assistant|>")[-1].strip().lower()
        
        # Clean and validate response
        response = re.sub(r'[^\w\s]', '', response).strip()
        
        # Find the category in the response
        for category in self.categories:
            if category in response:
                return category
        
        # Fallback classification based on keywords
        return self.fallback_classification(question)
    
    def fallback_classification(self, question):
        """
        Fallback classification using keyword matching
        """
        question_lower = question.lower()
        
        # Math keywords
        math_keywords = ['calculate', 'solve', 'equation', 'algebra', 'geometry', 'trigonometry', 
                        'calculus', 'derivative', 'integral', 'matrix', 'probability', 'statistics',
                        'number', 'sum', 'product', 'ratio', 'percentage', 'fraction']
        
        # Physics keywords
        physics_keywords = ['force', 'velocity', 'acceleration', 'momentum', 'energy', 'power',
                           'electric', 'magnetic', 'quantum', 'wave', 'frequency', 'mass',
                           'gravity', 'pressure', 'temperature', 'heat', 'light', 'motion']
        
        # Biology keywords
        biology_keywords = ['cell', 'dna', 'gene', 'organism', 'evolution', 'ecology',
                           'anatomy', 'physiology', 'bacteria', 'virus', 'protein',
                           'enzyme', 'metabolism', 'reproduction', 'species']
        
        # Count keyword matches
        math_score = sum(1 for keyword in math_keywords if keyword in question_lower)
        physics_score = sum(1 for keyword in physics_keywords if keyword in question_lower)
        biology_score = sum(1 for keyword in biology_keywords if keyword in question_lower)
        
        # Return category with highest score
        scores = {
            'math': math_score,
            'physics': physics_score,
            'biology': biology_score,
            'science': 0  # Default fallback
        }
        
        max_category = max(scores, key=scores.get)
        return max_category if scores[max_category] > 0 else 'science'
    
    def classify_batch(self, questions):
        """
        Classify multiple questions at once
        """
        results = []
        for question in questions:
            category = self.classify_question(question)
            results.append({
                'question': question,
                'category': category
            })
        return results

def main():
    """
    Main function to demonstrate the question classifier
    """
    print("Initializing Question Classifier...")
    classifier = QuestionClassifier()
    
    # Example questions for testing
    test_questions = [
        "What is the derivative of x^2?",
        "How does photosynthesis work?",
        "What is Newton's second law?",
        "What is the atomic structure of carbon?",
        "How do you solve a quadratic equation?",
        "What are mitochondria?",
        "Calculate the force needed to accelerate a 10kg object at 5 m/s²",
        "What is the process of cell division?"
    ]
    
    print("\nClassifying questions...\n")
    
    # Classify each question
    for question in test_questions:
        category = classifier.classify_question(question)
        print(f"Question: {question}")
        print(f"Category: {category}")
        print("-" * 50)
    
    # Interactive mode
    print("\nInteractive Mode (type 'quit' to exit):")
    while True:
        user_question = input("\nEnter your question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_question:
            category = classifier.classify_question(user_question)
            print(f"Classification: {category}")

if __name__ == "__main__":
    main()