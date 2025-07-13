import time
import torch
import re
import json
from transformers import AutoTokenizer, AutoProcessor, Llama4ForConditionalGeneration
from diffusers import DiffusionPipeline
from PIL import Image
import os
from datetime import datetime

class MathSolutionGenerator:
    def __init__(self, load_image_models=True):
        """Initialize the solution generator with LLaMA and SDXL models"""
        
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
        
        # Initialize Stable Diffusion XL models
        self.base_model = None
        self.refiner_model = None
        
        if load_image_models:
            self._load_sdxl_models()
    
    def _load_sdxl_models(self):
        """Load Stable Diffusion XL base and refiner models"""
        print("Loading Stable Diffusion XL models...")
        
        try:
            # Load base model
            print("Loading SDXL base model...")
            self.base_model = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                torch_dtype=torch.float16, 
                variant="fp16", 
                use_safetensors=True
            )
            self.base_model.to("cuda")
            
            # Load refiner model
            print("Loading SDXL refiner model...")
            self.refiner_model = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.base_model.text_encoder_2,
                vae=self.base_model.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.refiner_model.to("cuda")
            
            print("SDXL models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading SDXL models: {e}")
            print("Image generation will be disabled.")
            self.base_model = None
            self.refiner_model = None
    
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
            "Focus on mathematical accuracy, clear labels, simple line drawings, black and white style like textbook diagrams. "
            "Include specific measurements, geometric shapes, and educational styling instructions.]\n"
            "**END_IMAGE_GENERATION**\n\n"
            
            "The image prompt should be detailed but focused on creating clean, educational mathematical diagrams."
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
    
    def enhance_prompt_for_math_diagrams(self, prompt):
        """Enhance the prompt to generate better mathematical diagrams"""
        enhanced_prompt = f"""
        {prompt}
        
        Style: Clean mathematical diagram, textbook illustration style, black and white line drawing, 
        no shading or 3D effects, clear labels and measurements, educational geometry diagram, 
        simple and precise, high contrast, vector-style illustration
        """
        return enhanced_prompt.strip()
    
    def generate_image_with_sdxl(self, prompt, filename=None):
        """Generate image using Stable Diffusion XL"""
        if self.base_model is None or self.refiner_model is None:
            print("SDXL models not loaded. Cannot generate image.")
            return None
            
        print(f"Generating image with Stable Diffusion XL...")
        print(f"Prompt: {prompt}")
        
        try:
            # Enhance prompt for better mathematical diagrams
            enhanced_prompt = self.enhance_prompt_for_math_diagrams(prompt)
            
            # Generation parameters
            n_steps = 40
            high_noise_frac = 0.8
            
            # Generate with base model
            print("Generating with base model...")
            image = self.base_model(
                prompt=enhanced_prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type="latent",
                guidance_scale=7.5,
            ).images
            
            # Refine with refiner model
            print("Refining with refiner model...")
            image = self.refiner_model(
                prompt=enhanced_prompt,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
                guidance_scale=7.5,
            ).images[0]
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"math_diagram_{timestamp}.png"
            
            # Save image
            image.save(filename)
            print(f"Image generated and saved as: {filename}")
            
            return filename
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def solve_math_problem(self, question, img_url=None, request_diagram=False, 
                          save_image=True, output_filename=None):
        """Main method to solve math problem with optional diagram generation"""
        
        # Step 1: Generate solution with LLaMA
        llama_response = self.generate_solution_with_llama(question, img_url, request_diagram)
        
        # Step 2: Check if image generation is needed
        image_prompt = self.extract_image_prompt(llama_response)
        
        result = {
            "solution": llama_response,
            "image_generated": False,
            "image_file": None,
            "image_prompt": image_prompt
        }
        
        if image_prompt and save_image:
            print("\nImage generation requested by LLaMA...")
            
            # Step 3: Generate image with SDXL
            image_file = self.generate_image_with_sdxl(image_prompt, output_filename)
            
            if image_file:
                result["image_generated"] = True
                result["image_file"] = image_file
        
        return result
    
    def generate_standalone_diagram(self, prompt, filename=None):
        """Generate a diagram from a direct prompt"""
        return self.generate_image_with_sdxl(prompt, filename)

# Example usage
def main():
    # Initialize the solution generator
    generator = MathSolutionGenerator(load_image_models=True)
    
    # Example 1: Text-only question with diagram request
    question1 = """From a point Q, the length of the tangent to a circle is 24 cm and the distance of Q from the centre is 25 cm. Find the radius of the circle."""
    
    print("=" * 60)
    print("SOLVING PROBLEM 1 (with diagram request)")
    print("=" * 60)
    
    result1 = generator.solve_math_problem(
        question=question1,
        request_diagram=True,  # Request a diagram
        output_filename="tangent_circle_problem.png"
    )
    
    print("\nSOLUTION:")
    print(result1["solution"])
    print(f"\nImage generated: {result1['image_generated']}")
    if result1["image_generated"]:
        print(f"Image saved as: {result1['image_file']}")
    
    # Example 2: Question with input image
    print("\n" + "=" * 60)
    print("SOLVING PROBLEM 2 (with input image)")
    print("=" * 60)
    
    # Replace with actual image URL if you have one
    img_url = None  # Set to your image URL if available
    question2 = """Calculate the area of a triangle with base 12 cm and height 8 cm. 
                   Show the calculation and provide a labeled diagram."""
    
    result2 = generator.solve_math_problem(
        question=question2,
        img_url=img_url,
        request_diagram=True,
        output_filename="triangle_area_problem.png"
    )
    
    print("\nSOLUTION:")
    print(result2["solution"])
    print(f"\nImage generated: {result2['image_generated']}")
    if result2["image_generated"]:
        print(f"Image saved as: {result2['image_file']}")
    
    # Example 3: Direct diagram generation
    print("\n" + "=" * 60)
    print("GENERATING STANDALONE DIAGRAM")
    print("=" * 60)
    
    diagram_prompt = """Mathematical diagram showing a right triangle with legs of 3 cm and 4 cm, 
                       and hypotenuse of 5 cm. Label all sides clearly. Simple black and white 
                       line drawing, textbook style."""
    
    diagram_file = generator.generate_standalone_diagram(
        diagram_prompt, 
        "right_triangle_diagram.png"
    )
    
    if diagram_file:
        print(f"Standalone diagram saved as: {diagram_file}")

if __name__ == "__main__":
    main()


# Simplified usage example
def quick_solve(question, img_url=None, request_diagram=False, output_filename=None):
    """Quick function to solve a single problem"""
    generator = MathSolutionGenerator(load_image_models=True)
    
    result = generator.solve_math_problem(
        question=question,
        img_url=img_url,
        request_diagram=request_diagram,
        output_filename=output_filename
    )
    
    return result

# Example of using the quick function:
# result = quick_solve(
#     "Find the area of a circle with radius 5 cm. Show the solution with a diagram.",
#     request_diagram=True,
#     output_filename="circle_area_solution.png"
# )
# print(result["solution"])
# if result["image_generated"]:
#     print(f"Diagram saved as: {result['image_file']}")