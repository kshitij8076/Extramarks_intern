import time
import torch
import openai
import requests
import re
import json
from transformers import AutoTokenizer, AutoProcessor, Llama4ForConditionalGeneration
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
        
    def create_llama_messages(self, question, img_url=None, request_diagram=False):
        """Create messages for LLaMA-4-Scout with enhanced prompt for diagram generation"""
        
        system_content = (
            "You are a mathematics expert. The user will provide a math question consisting of an image and/or text description. "
            "Carefully analyze any provided image to extract necessary mathematical information (lengths, angles, geometry, diagrams, etc.). "
            "Then, use this information along with the text to solve the math question step-by-step. Show all workings clearly and compute the final answer. "
            
            "IMPORTANT: If the solution would benefit from a visual diagram or if the user specifically requests an output diagram, "
            "you MUST include a section at the end of your response with the following format:\n"
            "**IMAGE_GENERATION_NEEDED**\n"
            "PROMPT: [Detailed description for generating a clear, educational diagram that would help visualize the solution. "
            "Include specific measurements, labels, geometric shapes, and styling instructions for a textbook-style diagram.]\n"
            "**END_IMAGE_GENERATION**\n\n"
            
            "The image prompt should be detailed enough for an AI image generator to create an accurate mathematical diagram."
        )
        
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
            
        user_content.append({"type": "text", "text": question})
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def generate_solution_with_llama(self, question, img_url=None, request_diagram=False):
        """Generate solution using LLaMA-4-Scout"""
        print("Generating solution with LLaMA-4-Scout...")
        
        messages = self.create_llama_messages(question, img_url, request_diagram)
        
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
    
    def solve_math_problem(self, question, img_url=None, request_diagram=False, save_image=True):
        """Main method to solve math problem with optional diagram generation"""
        
        # Step 1: Generate solution with LLaMA
        llama_response = self.generate_solution_with_llama(question, img_url, request_diagram)
        
        # Step 2: Check if image generation is needed
        image_prompt = self.extract_image_prompt(llama_response)
        
        result = {
            "solution": llama_response,
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

# Example usage
def main():
    # Initialize the solution generator
    openai_api_key = "your-openai-api-key-here"  # Replace with your actual API key
    generator = MathSolutionGenerator(openai_api_key)
    
    # Example 1: Text-only question with diagram request
    question1 = """From a point Q, the length of the tangent to a circle is 24 cm and the distance of Q from the centre is 25 cm. Find the radius of the circle."""
    
    print("=" * 60)
    print("SOLVING PROBLEM 1 (with diagram request)")
    print("=" * 60)
    
    result1 = generator.solve_math_problem(
        question=question1,
        request_diagram=True  # Request a diagram
    )
    
    print("\nSOLUTION:")
    print(result1["solution"])
    print(f"\nImage generated: {result1['image_generated']}")
    if result1["image_generated"]:
        print(f"Image URL: {result1['image_url']}")
        print(f"Image saved as: {result1['image_file']}")
    
    # Example 2: Question with input image
    print("\n" + "=" * 60)
    print("SOLVING PROBLEM 2 (with input image)")
    print("=" * 60)
    
    # Replace with actual image URL
    img_url = "https://example.com/geometry-problem.jpg"
    question2 = "Analyze the given circle diagram and find the area of the shaded region. Show the calculation steps and provide a clearer diagram with labels."
    
    result2 = generator.solve_math_problem(
        question=question2,
        img_url=img_url,
        request_diagram=True
    )
    
    print("\nSOLUTION:")
    print(result2["solution"])
    print(f"\nImage generated: {result2['image_generated']}")
    if result2["image_generated"]:
        print(f"Image URL: {result2['image_url']}")
        print(f"Image saved as: {result2['image_file']}")

if __name__ == "__main__":
    main()


# Simplified usage example
def quick_solve(question, img_url=None, request_diagram=False):
    """Quick function to solve a single problem"""
    openai_api_key = "your-openai-api-key-here"  # Replace with your actual API key
    generator = MathSolutionGenerator(openai_api_key)
    
    result = generator.solve_math_problem(
        question=question,
        img_url=img_url,
        request_diagram=request_diagram
    )
    
    return result

# Example of using the quick function:
# result = quick_solve(
#     "Find the area of a circle with radius 5 cm. Show the solution with a diagram.",
#     request_diagram=True
# )
# print(result["solution"])