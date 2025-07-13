from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextStreamer
from qwen_vl_utils import process_vision_info
import torch
import re
import time

class MathStreamer(TextStreamer):
    """
    Custom streamer for real-time math problem solving with clean formatting
    """
    def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.current_text = ""
        self.buffer = ""
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Process and display text as it's being generated"""
        self.current_text += text
        self.buffer += text
        
        # Process complete sentences or mathematical expressions
        if any(char in text for char in ['.', ':', '\n', '=']) or stream_end:
            cleaned_chunk = self.clean_text_realtime(self.buffer)
            print(cleaned_chunk, end='', flush=True)
            self.buffer = ""
        
        if stream_end:
            print("\n\n" + "="*60)
            print("FINAL FORMATTED SOLUTION")
            print("="*60)
            final_formatted = format_complete_solution(self.current_text)
            print(final_formatted)
    
    def clean_text_realtime(self, text_chunk):
        """Clean text chunks in real-time"""
        # Remove LaTeX formatting
        cleaned = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text_chunk)
        cleaned = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', cleaned)
        cleaned = re.sub(r'\$([^$]*)\$', r'\1', cleaned)
        cleaned = re.sub(r'\\boxed\{([^}]+)\}', r'【\1】', cleaned)
        
        # Convert LaTeX symbols to Unicode
        symbol_replacements = {
            '\\theta': 'θ', '\\pi': 'π', '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ',
            '\\delta': 'δ', '\\omega': 'ω', '\\mu': 'μ', '\\sigma': 'σ', '\\Delta': 'Δ',
            '\\Sigma': 'Σ', '\\infty': '∞', '\\pm': '±', '\\times': '×', '\\div': '÷',
            '\\leq': '≤', '\\geq': '≥', '\\neq': '≠', '\\approx': '≈', '\\cdot': '·'
        }
        
        for latex_symbol, unicode_symbol in symbol_replacements.items():
            cleaned = cleaned.replace(latex_symbol, unicode_symbol)
        
        # Remove remaining backslashes and clean up
        cleaned = re.sub(r'\\[a-zA-Z]+', '', cleaned)
        cleaned = re.sub(r'\\', '', cleaned)
        
        return cleaned

def clean_and_format_response(raw_output):
    """
    Comprehensive cleaning and formatting of the model output
    """
    # Handle list input from model output
    if isinstance(raw_output, list) and len(raw_output) > 0:
        response = raw_output[0]
    else:
        response = str(raw_output)
    
    # Remove LaTeX formatting
    response = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', response)
    response = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', response)
    response = re.sub(r'\\\[([^\\]+)\\\]', r'\1', response)
    response = re.sub(r'\$\$([^$]+)\$\$', r'\1', response)
    response = re.sub(r'\$([^$]+)\$', r'\1', response)
    response = re.sub(r'\\boxed\{([^}]+)\}', r'【\1】', response)
    
    # Replace LaTeX symbols with Unicode
    symbol_replacements = {
        '\\theta': 'θ', '\\pi': 'π', '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ',
        '\\delta': 'δ', '\\omega': 'ω', '\\mu': 'μ', '\\sigma': 'σ', '\\Delta': 'Δ',
        '\\Sigma': 'Σ', '\\infty': '∞', '\\pm': '±', '\\times': '×', '\\div': '÷',
        '\\leq': '≤', '\\geq': '≥', '\\neq': '≠', '\\approx': '≈', '\\cdot': '·'
    }
    
    for latex_symbol, unicode_symbol in symbol_replacements.items():
        response = response.replace(latex_symbol, unicode_symbol)
    
    # Clean up remaining LaTeX commands
    response = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', response)
    response = re.sub(r'\\[a-zA-Z]+', '', response)
    response = re.sub(r'\\', '', response)
    
    # Clean up extra whitespace
    response = re.sub(r'\n\s*\n', '\n\n', response)
    response = response.strip()
    
    return response

def format_complete_solution(response):
    """
    Format the cleaned response into structured step-by-step solution
    """
    cleaned_response = clean_and_format_response(response)
    
    # Split into sections
    sections = cleaned_response.split('\n\n')
    formatted_solution = ""
    
    step_counter = 1
    in_solution = False
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # Detect different types of sections
        section_lower = section.lower()
        
        if any(keyword in section_lower for keyword in ['understanding', 'problem', 'given']):
            formatted_solution += f"### Step {step_counter}: Understanding the Problem\n"
            formatted_solution += f"{section}\n\n"
            step_counter += 1
            in_solution = True
            
        elif any(keyword in section_lower for keyword in ['concept', 'formula', 'principle', 'theory']):
            formatted_solution += f"### Step {step_counter}: Key Concepts and Formulas\n"
            formatted_solution += f"{section}\n\n"
            step_counter += 1
            
        elif any(keyword in section_lower for keyword in ['calculate', 'solve', 'find', 'determine']):
            formatted_solution += f"### Step {step_counter}: Calculations\n"
            formatted_solution += f"{section}\n\n"
            step_counter += 1
            
        elif any(keyword in section_lower for keyword in ['final answer', 'answer', 'result']):
            formatted_solution += f"### Final Answer\n"
            formatted_solution += f"{section}\n\n"
            
        else:
            if in_solution:
                formatted_solution += f"### Step {step_counter}: Solution Continuation\n"
                formatted_solution += f"{section}\n\n"
                step_counter += 1
            else:
                formatted_solution += f"{section}\n\n"
    
    return formatted_solution

def main():
    """
    Main function to run the math problem solver with streaming
    """
    print("Loading model...")
    
    # Load the model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/QVQ-72B-Preview", 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview")
    
    # Enhanced system prompt for structured solutions
    system_content = (
        "You are a mathematics expert providing clear, step-by-step solutions. "
        "Analyze any provided image to extract mathematical information, then solve systematically.\n\n"
        
        "SOLUTION STRUCTURE REQUIREMENTS:\n"
        "1. Start with 'Understanding the Problem' - state what is given and what needs to be found\n"
        "2. Follow with 'Key Concepts and Formulas' - mention relevant principles\n"
        "3. Provide detailed 'Calculations' with clear steps\n"
        "4. End with 'Final Answer' clearly highlighted\n\n"
        
        "FORMATTING REQUIREMENTS:\n"
        "- Use ONLY plain text mathematical notation - NO LaTeX\n"
        "- Use symbols: ×, ÷, √, π, θ, ω, α, β, γ, δ, μ, σ, Δ, Σ, ∞, ±, ≤, ≥, ≠, ≈\n"
        "- Format fractions as: (numerator)/(denominator)\n"
        "- Format exponents as: x^2, x^(n+1)\n"
        "- Format subscripts as: v_0, F_net, a_x\n"
        "- Show all mathematical work clearly\n"
        "- Explain each step thoroughly for educational value\n\n"
        
        "Make your response educational and easy to follow."
    )
    
    # Problem setup
    img_url = "/home/mukesh/extramarks/phy.jpg"
    question = "A simple pendulum with bob of mass m and conducting wire of length L swings under gravity through an angle 2θ​. The earth's magnetic field component in the direction perpendicular to swing is B. Calculate the maximum induced emf across the pendulum?"
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_content}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_url},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Process inputs
    print("Processing inputs...")
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
    
    # Create streamer
    streamer = MathStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    print("\n" + "="*60)
    print("STREAMING MATH SOLUTION")
    print("="*60)
    print("## Step-by-Step Solution\n")
    
    # Generate with streaming
    try:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.3,
            do_sample=True,
            streamer=streamer,
            pad_token_id=processor.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        # Save complete solution to file
        if hasattr(streamer, 'current_text') and streamer.current_text:
            with open('math_solution.txt', 'w', encoding='utf-8') as f:
                complete_solution = format_complete_solution(streamer.current_text)
                f.write(complete_solution)
            print(f"\n{'='*60}")
            print("Solution saved to 'math_solution.txt'")
            print("="*60)
            
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

if __name__ == "__main__":
    main()