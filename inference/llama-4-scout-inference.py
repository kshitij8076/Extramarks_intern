import time
import torch
import openai
import requests
import re
import json
from transformers import AutoTokenizer, AutoProcessor, Llama4ForConditionalGeneration, TextStreamer
from PIL import Image
import base64
from io import BytesIO

class MathSolutionGenerator:
    def __init__(self, openai_api_key):
        """Initialize the solution generator with both LLaMA and OpenAI models"""
        
        # Initialize LLaMA-4-Scout
        self.model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        print("Loading LLaMA-4-Scout model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_id,
            attn_implementation="sdpa",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("LLaMA-4-Scout model loaded successfully!")
        
        # Initialize OpenAI
        openai.api_key = openai_api_key
        
    def create_llama_messages(self, question, img_url=None, request_diagram=False, output_format="plain_math"):
        """Create messages for LLaMA-4-Scout with enhanced prompt for different output formats"""
        
        # Define different system prompts based on output format
        # if output_format == "plain_math":
        # system_content = (
        #     "You are a mathematics expert. The user will provide a math question consisting of an image and/or text description. "
        #     "Carefully analyze any provided image to extract necessary mathematical information (lengths, angles, geometry, diagrams, etc.). "
        #     "Then, use this information along with the text to solve the math question step-by-step. "

        #     "CRITICAL FORMATTING INSTRUCTIONS:\n"
        #     "- DO NOT use LaTeX formatting (no $, \\boxed{}, \\frac, \\sqrt, \\[\\], etc.)\n"
        #     "- DO NOT use Markdown formatting (no bold, italics, headers, or lists)\n"
        #     "- DO NOT wrap any math in dollar signs or LaTeX commands\n"
        #     "- DO NOT include any visual box like \\boxed{}\n"
        #     "- Present all output strictly in plain text and wrap the full response inside triple backticks (```), like a code block\n\n"

        #     "USE ONLY PLAIN TEXT SYMBOLS:\n"
        #     "- Use: ×, ÷, √, π, θ, ω, α, β, γ, δ, μ, σ, Δ, Σ, ∞, ±, ≤, ≥, ≠, ≈\n"
        #     "- Fractions: (numerator)/(denominator)\n"
        #     "- Exponents: x^2, a^(n+1)\n"
        #     "- Subscripts: v_0, F_net, a_x\n"
        #     "- Examples: v = u + at, F = ma, E = mc^2, A = πr^2\n"

        #     "- Show all steps and workings clearly.\n"
        #     "- Compute and clearly state the final answer.\n\n"

        #     "IMPORTANT: If the solution would benefit from a visual diagram, or the user specifically requests one, include the following at the end:\n"
        #     "**IMAGE_GENERATION_NEEDED**\n"
        #     "PROMPT: [Detailed description for generating a clear, educational diagram that would help visualize the solution. "
        #     "Include specific measurements, labels, geometric shapes, and styling instructions for a textbook-style diagram.]\n"
        #     "**END_IMAGE_GENERATION**\n\n"

        #     "REMEMBER: All content should be returned inside triple backticks so it is treated as plain preformatted text."
        # )

        # elif output_format == "latex":
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
        # else:  # structured format
        #     system_content = (
        #         "You are a mathematics expert. The user will provide a math question consisting of an image and/or text description. "
        #         "Carefully analyze any provided image to extract necessary mathematical information (lengths, angles, geometry, diagrams, etc.). "
        #         "Then, use this information along with the text to solve the math question step-by-step with clear headings and detailed explanations. "
        #         "Show all workings clearly and compute the final answer.\n\n"
                
        #         "IMPORTANT: If the solution would benefit from a visual diagram or if the user specifically requests an output diagram, "
        #         "you MUST include a section at the end of your response with the following format:\n"
        #         "**IMAGE_GENERATION_NEEDED**\n"
        #         "PROMPT: [Detailed description for generating a clear, educational diagram that would help visualize the solution. "
        #         "Include specific measurements, labels, geometric shapes, and styling instructions for a textbook-style diagram.]\n"
        #         "**END_IMAGE_GENERATION**\n\n"
                
        #         "The image prompt should be detailed enough for an AI image generator to create an accurate mathematical diagram."
        #     )
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_content}]
            }
        ]
        
        # Add user message
        user_content = []
        if img_url:
            user_content.append({"type": "image", "url": img_url})
        
        # Modify question to request diagram if needed
        if request_diagram:
            question += "\n\nPlease also provide an output diagram to visualize the solution."
        
        # Add format-specific instructions to the question
        if output_format == "plain_math":
            question += "\n\nPlease write all mathematical expressions in plain text format (like F = ma, v = √(2gh), etc.) without any LaTeX formatting."
            
        user_content.append({"type": "text", "text": question})
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def generate_solution_with_llama(self, question, img_url=None, request_diagram=False, output_format="plain_math"):
        """Generate solution using LLaMA-4-Scout"""
        print("Generating solution with LLaMA-4-Scout...")
        
        messages = self.create_llama_messages(question, img_url, request_diagram, output_format)
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.7,
            do_sample=True,
        )
        
        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )[0]
        
        return response
    
    def generate_solution_streaming(self, question, img_url=None, request_diagram=False, 
                                  output_format="plain_math", max_new_tokens=1500, 
                                  temperature=0.7, random_seed=42):
        """Generate solution with streaming output"""
        print("Generating solution with streaming output...")
        
        messages = self.create_llama_messages(question, img_url, request_diagram, output_format)
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Set random seed for reproducibility
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Create text streamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        print(f"\nUser: {question}")
        print(f"Assistant: ", end="")
        
        # Generate with streaming
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                streamer=streamer,
                use_cache=True,
                repetition_penalty=1.05,
                top_p=0.95,
                top_k=50,
            )
        
        # Return the full response for further processing
        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )[0]
        
        return response
    
    def generate_solution_streaming_plain_math(self, question, img_url=None, request_diagram=False, 
                                             max_new_tokens=1500):
        """Generate solution with plain text math formatting and streaming"""
        return self.generate_solution_streaming(
            question=question,
            img_url=img_url,
            request_diagram=request_diagram,
            output_format="plain_math",
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for consistency
            random_seed=42
        )
    
    def generate_solution_streaming_deterministic(self, question, img_url=None, request_diagram=False, 
                                                max_new_tokens=1500, output_format="plain_math"):
        """Generate completely deterministic solution with streaming"""
        return self.generate_solution_streaming(
            question=question,
            img_url=img_url,
            request_diagram=request_diagram,
            output_format=output_format,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # No randomness
            random_seed=42
        )
    
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
    
    def generate_image_with_dalle(self, prompt):
        """Generate image using DALL-E 3"""
        print(f"Generating image with DALL-E 3...")
        print(f"Prompt: {prompt}")
        
        try:
            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard"
            )
            
            image_url = response.data[0].url
            print(f"Image generated successfully: {image_url}")
            return image_url
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def download_and_save_image(self, image_url, filename="generated_diagram.png"):
        """Download and save the generated image"""
        try:
            image_data = requests.get(image_url).content
            with open(filename, "wb") as f:
                f.write(image_data)
            print(f"Image saved as {filename}")
            return filename
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def solve_math_problem_streaming(self, question, img_url=None, request_diagram=False, 
                                   output_format="plain_math", save_image=True):
        """Main method to solve math problem with streaming output and optional diagram generation"""
        
        # Step 1: Generate solution with streaming
        raw_response = self.generate_solution_streaming(
            question, img_url, request_diagram, output_format
        )
        
        # Step 2: Format the response for better readability
        if output_format == "plain_math":
            formatted_response = self.format_solution_output(raw_response)
        else:
            formatted_response = raw_response
        
        # Step 3: Check if image generation is needed
        image_prompt = self.extract_image_prompt(raw_response)
        
        result = {
            "solution": formatted_response,
            "image_generated": False,
            "image_url": None,
            "image_file": None
        }
        
        if image_prompt:
            print("\n\nImage generation requested by LLaMA...")
            
            # Step 4: Generate image with DALL-E
            image_url = self.generate_image_with_dalle(image_prompt)
            
            if image_url:
                result["image_generated"] = True
                result["image_url"] = image_url
                
                # Step 5: Save image if requested
                if save_image:
                    filename = self.download_and_save_image(image_url)
                    result["image_file"] = filename
        
        return result
    
    def solve_math_problem(self, question, img_url=None, request_diagram=False, save_image=True, output_format="plain_math"):
        """Main method to solve math problem with optional diagram generation (non-streaming)"""
        
        # Step 1: Generate solution with LLaMA
        raw_response = self.generate_solution_with_llama(question, img_url, request_diagram, output_format)
    
        # Format the response for better readability
        if output_format == "plain_math":
            formatted_response = self.format_solution_output(raw_response)
        else:
            formatted_response = raw_response
        
        # Step 2: Check if image generation is needed
        image_prompt = self.extract_image_prompt(raw_response)
        
        result = {
            "solution": formatted_response,
            "image_generated": False,
            "image_url": None,
            "image_file": None
        }
        
        if image_prompt:
            print("\nImage generation requested by LLaMA...")
            
            # Step 3: Generate image with DALL-E
            image_url = self.generate_image_with_dalle(image_prompt)
            
            if image_url:
                result["image_generated"] = True
                result["image_url"] = image_url
                
                # Step 4: Save image if requested
                if save_image:
                    filename = self.download_and_save_image(image_url)
                    result["image_file"] = filename
        
        return result
    
    def enhanced_latex_to_plain_text(self, text):
        """Enhanced conversion from LaTeX to plain text mathematical expressions"""
        
        replacements = {
            # Complex fractions first
            r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}': r'(\1)/(\2)',
            
            # Square roots
            r'\\sqrt\{([^}]+)\}': r'√(\1)',
            r'\\sqrt\[([^]]+)\]\{([^}]+)\}': r'(\2)^(1/\1)',
            
            # Greek letters (lowercase)
            r'\\alpha\b': 'α', r'\\beta\b': 'β', r'\\gamma\b': 'γ', r'\\delta\b': 'δ',
            r'\\epsilon\b': 'ε', r'\\varepsilon\b': 'ε', r'\\zeta\b': 'ζ', r'\\eta\b': 'η',
            r'\\theta\b': 'θ', r'\\vartheta\b': 'θ', r'\\iota\b': 'ι', r'\\kappa\b': 'κ',
            r'\\lambda\b': 'λ', r'\\mu\b': 'μ', r'\\nu\b': 'ν', r'\\xi\b': 'ξ',
            r'\\omicron\b': 'ο', r'\\pi\b': 'π', r'\\varpi\b': 'π', r'\\rho\b': 'ρ',
            r'\\varrho\b': 'ρ', r'\\sigma\b': 'σ', r'\\varsigma\b': 'ς', r'\\tau\b': 'τ',
            r'\\upsilon\b': 'υ', r'\\phi\b': 'φ', r'\\varphi\b': 'φ', r'\\chi\b': 'χ',
            r'\\psi\b': 'ψ', r'\\omega\b': 'ω',
            
            # Greek letters (uppercase)
            r'\\Alpha\b': 'Α', r'\\Beta\b': 'Β', r'\\Gamma\b': 'Γ', r'\\Delta\b': 'Δ',
            r'\\Epsilon\b': 'Ε', r'\\Zeta\b': 'Ζ', r'\\Eta\b': 'Η', r'\\Theta\b': 'Θ',
            r'\\Iota\b': 'Ι', r'\\Kappa\b': 'Κ', r'\\Lambda\b': 'Λ', r'\\Mu\b': 'Μ',
            r'\\Nu\b': 'Ν', r'\\Xi\b': 'Ξ', r'\\Omicron\b': 'Ο', r'\\Pi\b': 'Π',
            r'\\Rho\b': 'Ρ', r'\\Sigma\b': 'Σ', r'\\Tau\b': 'Τ', r'\\Upsilon\b': 'Υ',
            r'\\Phi\b': 'Φ', r'\\Chi\b': 'Χ', r'\\Psi\b': 'Ψ', r'\\Omega\b': 'Ω',
            
            # Mathematical operators
            r'\\times\b': '×', r'\\cdot\b': '·', r'\\div\b': '÷', r'\\pm\b': '±',
            r'\\mp\b': '∓', r'\\ast\b': '*', r'\\star\b': '⋆', r'\\circ\b': '∘',
            r'\\bullet\b': '•', r'\\oplus\b': '⊕', r'\\ominus\b': '⊖', r'\\otimes\b': '⊗',
            r'\\oslash\b': '⊘', r'\\odot\b': '⊙',
            
            # Relations
            r'\\neq\b': '≠', r'\\ne\b': '≠', r'\\leq\b': '≤', r'\\le\b': '≤',
            r'\\geq\b': '≥', r'\\ge\b': '≥', r'\\ll\b': '≪', r'\\gg\b': '≫',
            r'\\equiv\b': '≡', r'\\approx\b': '≈', r'\\sim\b': '∼', r'\\simeq\b': '≃',
            r'\\cong\b': '≅', r'\\propto\b': '∝', r'\\parallel\b': '∥', r'\\perp\b': '⊥',
            
            # Special symbols
            r'\\infty\b': '∞', r'\\partial\b': '∂', r'\\nabla\b': '∇', r'\\sum\b': 'Σ',
            r'\\prod\b': 'Π', r'\\int\b': '∫', r'\\oint\b': '∮',
            
            # Superscripts and subscripts
            r'\^\{([^}]+)\}': r'^(\1)', r'_\{([^}]+)\}': r'_(\1)',
            r'\^([a-zA-Z0-9])': r'^\1', r'_([a-zA-Z0-9])': r'_\1',
            
            # Remove LaTeX delimiters
            r'\\\[': '', r'\\\]': '', r'\\\(': '', r'\\\)': '',
            r'\$\$': '', r'\$': '',
            
            # Remove boxed expressions
            r'\\boxed\{([^}]+)\}': r'**\1**',
            
            # Clean up spacing
            r'\s+': ' ',
        }
        
        formatted_text = text
        for pattern, replacement in replacements.items():
            formatted_text = re.sub(pattern, replacement, formatted_text)
        
        # Clean up any remaining backslashes and braces
        formatted_text = re.sub(r'\\[a-zA-Z]+', '', formatted_text)
        formatted_text = re.sub(r'[{}]', '', formatted_text)
        
        return formatted_text.strip()
    
    def format_solution_output(self, raw_solution):
        """Format the raw solution for better readability"""
        # Apply enhanced LaTeX to plain text conversion
        formatted = self.enhanced_latex_to_plain_text(raw_solution)
        
        # Add better spacing around equations
        formatted = re.sub(r'([a-zA-Z])\s*=\s*', r'\1 = ', formatted)
        formatted = re.sub(r'=\s*([a-zA-Z])', r'= \1', formatted)
        
        # Bold important final answers
        formatted = re.sub(r'(Final Answer:?\s*[^.\n]+)', r'**\1**', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'(Answer:?\s*[^.\n]+)', r'**\1**', formatted, flags=re.IGNORECASE)
        
        return formatted
    
    def chat_mode_streaming(self):
        """Interactive chat mode with streaming output"""
        print("\n=== Interactive Math Chat Mode (Streaming) ===")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'format:latex' to switch to LaTeX format")
        print("Type 'format:plain' to switch to plain text format")
        print("Type 'diagram:on' to request diagrams automatically")
        print("Type 'diagram:off' to disable automatic diagrams")
        print("-" * 50)
        
        output_format = "plain_math"
        auto_diagram = False
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'format:latex':
                output_format = "latex"
                print("Switched to LaTeX format")
                continue
            elif user_input.lower() == 'format:plain':
                output_format = "plain_math"
                print("Switched to plain text format")
                continue
            elif user_input.lower() == 'diagram:on':
                auto_diagram = True
                print("Auto-diagram enabled")
                continue
            elif user_input.lower() == 'diagram:off':
                auto_diagram = False
                print("Auto-diagram disabled")
                continue
            
            if not user_input:
                continue
            
            # Process the math question with streaming
            self.solve_math_problem_streaming(
                question=user_input,
                request_diagram=auto_diagram,
                output_format=output_format
            )


# Example usage
def main():
    # Initialize the solution generator
    openai_api_key = "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"
    generator = MathSolutionGenerator(openai_api_key)
    
    # Example 1: Text-only question with plain math streaming
    question1 = """Define the term 'mobility' of charge carriers in a current carrying conductor. Obtain the relation for mobility in terms of relaxation time."""
    
    print("=" * 60)
    print("SOLVING PROBLEM 1 (Plain Math Streaming)")
    print("=" * 60)
    
    # Example 2: Question with input image + output diagram
    img_url = "/home/mukesh/extramarks/Screenshot 2025-05-30 102053.png"
    question2 = "A simple pendulum with bob of mass m and conducting wire of length L swings under gravity through an angle 2θ​. The earth's magnetic field component in the direction perpendicular to swing is B.Calculate the maximum induced emf across the pendulum?"
    question3 = """The angle of elevation of the top P
 of a tower from the feet of one person standing due South of the tower is 45deg
 and from the feet of another person standing due west of the tower is 30deg
. If the height of the tower is 5 meters, then the distance (in meters) between the two persons is equal to"""
    result1 = generator.solve_math_problem_streaming(
        question=question3,
        # img_url=img_url,
        request_diagram=False,
        output_format="latex"
    )
    
    print(f"\nImage generated: {result1['image_generated']}")
    if result1["image_generated"]:
        print(f"Image URL: {result1['image_url']}")
        print(f"Image saved as: {result1['image_file']}")
    
 
    
    # question2 = "Find the electric field at a point P located at distance r from a point charge Q. Show all derivation steps."
    
    # result2 = generator.generate_solution_streaming_plain_math(
    #     question=question2,
    #     request_diagram=True
    # )
    
    # Example 3: Interactive chat mode
    # print("\n" + "=" * 60)
    # print("STARTING INTERACTIVE CHAT MODE")
    # print("=" * 60)
    # generator.chat_mode_streaming()

if __name__ == "__main__":
    main()

