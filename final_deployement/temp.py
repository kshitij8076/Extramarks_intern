import base64
import requests
from openai import OpenAI
from typing import Optional, Union, Dict, Tuple
from pathlib import Path
import re

class GPT4OVisionHandler:
    def __init__(self, api_key: str):
        """
        Initialize the GPT-4o Vision handler with your OpenAI API key.
        
        Args:
            api_key: Your OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"  # or "gpt-4o-mini" for a faster, cheaper option
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def clean_latex_for_terminal(self, text: str) -> str:
        """
        Convert LaTeX formatting to plain text for terminal display.
        
        Args:
            text: Text containing LaTeX formatting
            
        Returns:
            Cleaned text suitable for terminal display
        """
        # Replace common LaTeX patterns
        replacements = {
            r'\(' : '(',
            r'\)' : ')',
            r'\[' : '\n',
            r'\]' : '\n',
            r'\{' : '{',
            r'\}' : '}',
            r'\\boxed{([^}]+)}': r'[ANSWER: \1]',
            r'\\frac{([^}]+)}{([^}]+)}': r'(\1/\2)',
            r'\\left\(': '(',
            r'\\right\)': ')',
            r'\\cdot': '·',
            r'\\times': '×',
            r'\\div': '÷',
            r'\\pm': '±',
            r'\\sqrt{([^}]+)}': r'√(\1)',
            r'\\sqrt': '√',
            r'\^{([^}]+)}': r'^(\1)',
            r'\_\{([^}]+)\}': r'_(\1)',
            r'\\alpha': 'α',
            r'\\beta': 'β',
            r'\\gamma': 'γ',
            r'\\delta': 'δ',
            r'\\theta': 'θ',
            r'\\pi': 'π',
            r'\\sum': 'Σ',
            r'\\int': '∫',
            r'\\infty': '∞',
            r'\\leq': '≤',
            r'\\geq': '≥',
            r'\\neq': '≠',
            r'\\approx': '≈',
            r'  +': ' ',  # Multiple spaces to single space
        }
        
        # Apply replacements
        cleaned = text
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Remove any remaining backslashes before letters (LaTeX commands)
        cleaned = re.sub(r'\\([a-zA-Z]+)', r'\1', cleaned)
        
        # Clean up extra newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned
    
    def answer_question(self, 
                       question: str, 
                       image_path: Optional[str] = None,
                       image_url: Optional[str] = None,
                       max_tokens: int = 1000,
                       use_custom_prompt: bool = True,
                       clean_latex: bool = True) -> str:
        """
        Answer a question using GPT-4o with the step-by-step solving prompt.
        
        Args:
            question: The question to answer
            image_path: Optional path to a local image file
            image_url: Optional URL to an image (alternative to image_path)
            max_tokens: Maximum tokens in the response
            use_custom_prompt: Use the step-by-step solving prompt (default: True)
            clean_latex: Clean LaTeX formatting for terminal display (default: True)
            
        Returns:
            The answer from GPT-4o
        """
        # Build the messages array
        messages = []
        
        if use_custom_prompt:
            # Use the specific solve prompt format
            system_content = "You are a helpful AI assistant. Solve the given question step by step and provide only the solution for the given question."
            messages.append({
                "role": "system",
                "content": system_content
            })
            
            # The user message with the formatted prompt
            user_prompt = f"{question}"
            
            # The assistant prefix to encourage step-by-step format
            assistant_prefix = "I'll solve this step by step:"
        else:
            # Just use the question directly
            user_prompt = question
            assistant_prefix = None
        
        # Create the content for the user message
        content = []
        
        # Add the text question
        content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # Add image if provided
        if image_path:
            # Encode local image
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        elif image_url:
            # Use image URL directly
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
        
        # Add the user message
        messages.append({
            "role": "user",
            "content": content
        })
        
        try:
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            
            # Get the response
            answer = response.choices[0].message.content
            
            # If using custom prompt and no assistant prefix was naturally included,
            # prepend it to maintain consistency
            if use_custom_prompt and assistant_prefix and not answer.startswith(assistant_prefix):
                answer = f"{assistant_prefix}\n\n{answer}"
            
            # Clean LaTeX if requested
            if clean_latex:
                answer = self.clean_latex_for_terminal(answer)
            
            return answer
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def classify_question(self, question: str, image_path: Optional[str] = None, image_url: Optional[str] = None) -> Dict[str, Union[int, str]]:
        """
        Classify a question by difficulty level (1-5) and subject category.
        
        Args:
            question: The question to classify
            image_path: Optional path to a local image file
            image_url: Optional URL to an image
            
        Returns:
            Dictionary with 'level' (1-5) and 'subject' keys
        """
        classification_prompt = f"""You are a question difficulty classifier. Your task is to assign a difficulty level from 1 to 5 to any given question based on its complexity, required concepts, and reasoning depth. Additionally, classify the subject category.

**Difficulty Levels:**

**Level 1 – Basic Arithmetic**
Questions involve direct, single-step calculations like basic addition, subtraction, multiplication, or division. No reasoning or multi-step logic required.
Examples:
- What is 6 + 4?
- Multiply 3 and 7.

**Level 2 – Elementary Problem Solving**
Slightly more involved than Level 1. May require interpreting short text, combining 2 steps, or applying elementary math (e.g., perimeter, averages). No abstract reasoning.
Examples:
- A pencil costs ₹5. How much do 3 pencils cost?
- What is the average of 10, 20, and 30?

**Level 3 – Moderate Conceptual Thinking**
Requires understanding of basic algebra, geometry, or logic. Involves multiple steps or concepts but each step is straightforward. Some reasoning is needed.
Examples:
- Solve for x: 2x + 3 = 11
- Find the area of a triangle with base 6 cm and height 4 cm.

**Level 4 – Advanced Reasoning / Multi-Step Logic**
Involves chaining multiple concepts together. Could include algebraic manipulation, conditional reasoning, or interpreting diagrams. May not be solvable in a single glance.
Examples:
- A train leaves station A at 9:00 AM and another from station B at 9:30 AM...
- If 4x – 2 = 3y and x + y = 10, what is x?

**Level 5 – Intense Multi-Concept Reasoning**
Requires deep understanding, abstraction, and combining multiple areas (e.g., number theory, combinatorics, geometry, logic). These are conceptually rich and potentially tricky, even for experienced students.
Examples:
- Given a function f(x) = 2x² + 3x + 1, find the smallest positive integer for which f(f(x)) = 0.
- Complex spatial and algebraic reasoning problems.

**Subject Categories:**
- **general**: General knowledge, everyday questions
- **maths**: Mathematics problems, calculations, equations, algebra, calculus, geometry, trigonometry, statistics, probability
- **physics**: Physics concepts like force, energy, motion, electricity, magnetism, thermodynamics, optics, waves
- **biology**: Living organisms, cells, DNA, genetics, anatomy, physiology, ecology, evolution
- **chemistry**: Chemical reactions, periodic table, elements, compounds, acids, bases, molecular structure
- **prover**: Mathematical proofs, theorem proving (questions asking to "prove", "show that", "demonstrate", "verify", "establish", "derive")

**Classification Rules:**
1. **prover**: If the question asks to "prove", "show that", "demonstrate", "verify", "establish", or "derive" something mathematical → classify as "prover"
2. Focus on the cognitive complexity and reasoning depth required
3. Consider the number of steps and concepts that need to be combined
4. Evaluate whether the solution is immediate or requires deeper thinking

**Question to classify:** {question}

**Your Output Format:**
Level: [1-5]
Subject: [subject_name]"""

        # Build messages
        messages = [
            {
                "role": "system",
                "content": "You are a question difficulty classifier. Analyze questions and classify them by difficulty and subject."
            }
        ]
        
        # Build content
        content = []
        content.append({
            "type": "text",
            "text": classification_prompt
        })
        
        # Add image if provided
        if image_path:
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        elif image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        try:
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=100  # Classification doesn't need many tokens
            )
            
            # Parse the response
            result_text = response.choices[0].message.content
            
            # Extract level and subject using regex
            level_match = re.search(r'Level:\s*(\d)', result_text)
            subject_match = re.search(r'Subject:\s*(\w+)', result_text)
            
            level = int(level_match.group(1)) if level_match else 0
            subject = subject_match.group(1).lower() if subject_match else "unknown"
            
            return {
                "level": level,
                "subject": subject,
                "raw_response": result_text
            }
            
        except Exception as e:
            return {
                "level": 0,
                "subject": "error",
                "raw_response": f"Error: {str(e)}"
            }
    
    def answer_and_classify(self, 
                          question: str, 
                          image_path: Optional[str] = None,
                          image_url: Optional[str] = None,
                          max_tokens: int = 1000,
                          clean_latex: bool = True) -> Dict[str, Union[str, int]]:
        """
        Answer a question and classify it in one go.
        
        Args:
            question: The question to answer and classify
            image_path: Optional path to a local image file
            image_url: Optional URL to an image
            max_tokens: Maximum tokens for the answer
            clean_latex: Clean LaTeX formatting for terminal display (default: True)
            
        Returns:
            Dictionary with 'answer', 'level', 'subject', and formatted output
        """
        # Get the answer
        answer = self.answer_question(question, image_path, image_url, max_tokens, clean_latex=clean_latex)
        
        # Get the classification
        classification = self.classify_question(question, image_path, image_url)
        
        # Create formatted output
        difficulty_descriptions = {
            1: "Basic Arithmetic",
            2: "Elementary Problem Solving",
            3: "Moderate Conceptual Thinking",
            4: "Advanced Reasoning / Multi-Step Logic",
            5: "Intense Multi-Concept Reasoning"
        }
        
        formatted_output = f"""
{'='*60}
QUESTION: {question}
{'='*60}

CLASSIFICATION:
- Difficulty Level: {classification['level']} ({difficulty_descriptions.get(classification['level'], 'Unknown')})
- Subject: {classification['subject'].title()}

ANSWER:
{answer}
{'='*60}
"""
        
        return {
            "question": question,
            "answer": answer,
            "level": classification['level'],
            "subject": classification['subject'],
            "formatted_output": formatted_output
        }
    
        """
        Answer multiple questions, each potentially with its own image.
        
        Args:
            questions_data: List of dictionaries with 'question' and optional 'image_path' or 'image_url'
            use_custom_prompt: Use the step-by-step solving prompt for all questions
            
        Returns:
            List of answers
        """
        answers = []
        
        for data in questions_data:
            question = data.get('question', '')
            image_path = data.get('image_path')
            image_url = data.get('image_url')
            
            answer = self.answer_question(question, image_path, image_url, use_custom_prompt=use_custom_prompt)
            answers.append({
                'question': question,
                'answer': answer
            })
        
        return answers

# Simple function approach with the solve prompt
def ask_gpt4o_with_solve_prompt(api_key: str, 
                                question: str, 
                                image_path: Optional[str] = None,
                                image_url: Optional[str] = None) -> str:
    """
    Ask GPT-4o a question using the step-by-step solving prompt.
    
    Args:
        api_key: Your OpenAI API key
        question: The question to ask
        image_path: Optional path to local image
        image_url: Optional URL to image
        
    Returns:
        The answer from GPT-4o
    """
    client = OpenAI(api_key=api_key)
    
    # Build messages with the solve prompt format
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Solve the given question step by step and provide only the solution for the given question."
        }
    ]
    
    # Build content
    content = [{"type": "text", "text": question}]
    
    if image_path:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    elif image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Simple function for classification
def classify_question_simple(api_key: str, 
                           question: str,
                           image_path: Optional[str] = None) -> Tuple[int, str]:
    """
    Simple function to classify a question's difficulty and subject.
    
    Args:
        api_key: Your OpenAI API key
        question: The question to classify
        image_path: Optional path to image
        
    Returns:
        Tuple of (difficulty_level, subject)
    """
    handler = GPT4OVisionHandler(api_key)
    result = handler.classify_question(question, image_path)
    return result['level'], result['subject']

# Example usage
def main():
    # Initialize with your API key
    api_key = "your-openai-api-key-here"
    handler = GPT4OVisionHandler(api_key)
    
    # Example 1: Basic classification
    print("Example 1: Classify a simple math problem")
    classification = handler.classify_question("What is 6 + 4?")
    print(f"Level: {classification['level']} (1-5)")
    print(f"Subject: {classification['subject']}")
    print(f"Raw response: {classification['raw_response']}\n")
    
    # Example 2: Answer and classify together
    print("Example 2: Answer and classify a moderate problem")
    result = handler.answer_and_classify("Solve for x: 2x + 3 = 11")
    print(result['formatted_output'])
    
    # Example 3: Complex problem with image
    print("Example 3: Complex problem with image")
    result = handler.answer_and_classify(
        "Prove that the sum of angles in the triangle shown equals 180 degrees",
        image_path="path/to/triangle.jpg"
    )
    print(result['formatted_output'])
    
    # Example 4: Multiple questions with classification
    print("Example 4: Multiple questions with classification")
    questions = [
        {"question": "What is 3 × 7?"},
        {"question": "A train travels 60 km/h for 2 hours. How far did it go?"},
        {"question": "Solve: x² + 5x + 6 = 0"},
        {"question": "Prove that √2 is irrational"},
        {"question": "What is the pH of a 0.001M HCl solution?"}
    ]
    
    results = handler.answer_multiple_questions(questions, classify=True)
    
    # Display results in a table format
    print("\n" + "="*80)
    print(f"{'Question':<40} {'Level':<8} {'Subject':<12}")
    print("="*80)
    for r in results:
        q = r['question'][:37] + "..." if len(r['question']) > 40 else r['question']
        print(f"{q:<40} {r['level']:<8} {r['subject']:<12}")
    print("="*80)
    
    # Example 5: Just classification without solving
    print("\nExample 5: Quick classification of multiple questions")
    quick_questions = [
        "What is 2+2?",
        "Calculate the derivative of x³",
        "Explain photosynthesis",
        "Balance: Fe + O₂ → Fe₂O₃"
    ]
    
    for q in quick_questions:
        c = handler.classify_question(q)
        print(f"Q: {q}")
        print(f"   Level: {c['level']}, Subject: {c['subject']}")

# Utility function to analyze a batch of questions
def analyze_question_batch(api_key: str, questions: list):
    """
    Analyze a batch of questions and provide statistics.
    
    Args:
        api_key: Your OpenAI API key
        questions: List of question strings
    """
    handler = GPT4OVisionHandler(api_key)
    
    classifications = []
    for q in questions:
        c = handler.classify_question(q)
        classifications.append(c)
    
    # Calculate statistics
    levels = [c['level'] for c in classifications]
    subjects = [c['subject'] for c in classifications]
    
    print("\n📊 BATCH ANALYSIS RESULTS")
    print("="*50)
    print(f"Total questions: {len(questions)}")
    print(f"Average difficulty: {sum(levels)/len(levels):.1f}")
    print(f"Difficulty distribution:")
    for level in range(1, 6):
        count = levels.count(level)
        bar = "█" * count
        print(f"  Level {level}: {bar} ({count})")
    
    print(f"\nSubject distribution:")
    subject_counts = {}
    for s in subjects:
        subject_counts[s] = subject_counts.get(s, 0) + 1
    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count}")
    print("="*50)

# Example showing the exact prompt structure sent to the API
def show_prompt_structure():
    """
    This shows exactly what gets sent to the OpenAI API with the solve prompt
    """
    example_api_payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Solve the given question step by step and provide only the solution for the given question."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is 1543 × 267?"
                    }
                    # Image would be added here if provided
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    print("This is the exact structure sent to the API:")
    print(example_api_payload)
    
    # Example for classification
    classification_example = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a question difficulty classifier. Analyze questions and classify them by difficulty and subject."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a question difficulty classifier... [full prompt]... Level: [1-5]\nSubject: [subject_name]"
                    }
                ]
            }
        ],
        "max_tokens": 100
    }
    
    print("\nFor classification, this structure is sent:")
    print("(Classification prompt truncated for brevity)")

if __name__ == "__main__":
    # Example usage
    API_KEY = "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"
    
    # Initialize handler
    handler = GPT4OVisionHandler(API_KEY)
    
    result = handler.answer_and_classify(
        "Solve the problem shown in this image", 
        image_path="/home/karang/dikshant/extramarks_239/q2.jpg"
    )
    print(result['formatted_output']) 