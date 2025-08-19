"""
Examples showing how vision-language models handle text extraction and classification
in a single unified approach.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json


class UnifiedVLProcessor:
    """
    Demonstrates the power of using VL models for both OCR and understanding.
    """
    
    def __init__(self, model_size="7B"):
        """Initialize with specified model size."""
        model_name = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def analyze_document(self, image_path: str, task: str = "classify") -> dict:
        """
        Analyze a document with different task prompts.
        
        Tasks:
        - "classify": Determine if visual elements are needed
        - "extract": Extract all text content
        - "summarize": Summarize the content
        - "qa": Answer questions about the content
        """
        
        task_prompts = {
            "classify": """Analyze this image comprehensively:
1. Extract all text content
2. Identify visual elements (diagrams, charts, tables, etc.)
3. Determine if questions about this content would need visual information

Format your response as:
TEXT_CONTENT: [all text from the image]
VISUAL_ELEMENTS: [list of visual elements]
CLASSIFICATION: [text_only if text is sufficient, needs_visual if visual elements are required]
EXAMPLE_TEXT_QUESTIONS: [questions that can be answered with text only]
EXAMPLE_VISUAL_QUESTIONS: [questions that need the visual elements]""",
            
            "extract": """Extract ALL text content from this image. Include:
- Main body text
- Headings and titles
- Labels on diagrams
- Text in tables
- Captions
- Any mathematical equations or formulas
- Numbers and data values

Be thorough and preserve the structure where possible.""",
            
            "summarize": """Provide a comprehensive summary of this document including:
1. Main topic/purpose
2. Key points or findings
3. Any data or statistics mentioned
4. Visual elements present and their purpose
5. Overall structure of the document""",
            
            "qa": """Analyze this document and identify:
1. What type of questions could be asked about this content?
2. Which questions would need visual information to answer?
3. Which questions could be answered from text alone?

Provide 3 examples of each type."""
        }
        
        prompt = task_prompts.get(task, task_prompts["classify"])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process through model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]
        
        return {
            "task": task,
            "output": output,
            "image": image_path
        }
    
    def smart_routing(self, image_path: str, user_question: str) -> dict:
        """
        Smart routing that considers both the document content and user question.
        """
        
        prompt = f"""Analyze this document and the user's question to determine the best model to use.

User's question: "{user_question}"

Consider:
1. What information from the document is needed to answer this question?
2. Is the visual layout, structure, or graphical elements necessary?
3. Can the question be answered using only the textual content?

Provide:
RELEVANT_CONTENT: [specific parts of the document relevant to the question]
VISUAL_DEPENDENCY: [explain if/why visual elements are needed]
ROUTING: [text_model OR vision_model]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation of routing decision]"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]
        
        return self._parse_routing_output(output, user_question)
    
    def _parse_routing_output(self, output: str, user_question: str) -> dict:
        """Parse the routing output into structured format."""
        
        result = {
            "question": user_question,
            "routing": "vision_model",  # default to safer option
            "confidence": 0.5,
            "reasoning": "",
            "relevant_content": "",
            "visual_dependency": ""
        }
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("ROUTING:"):
                routing = line[8:].strip().lower()
                result["routing"] = "text_model" if "text_model" in routing else "vision_model"
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line[11:].strip())
                except:
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line[10:].strip()
            elif line.startswith("RELEVANT_CONTENT:"):
                result["relevant_content"] = line[17:].strip()
            elif line.startswith("VISUAL_DEPENDENCY:"):
                result["visual_dependency"] = line[18:].strip()
        
        return result


# Example usage scenarios
def example_scenarios():
    """Demonstrate various usage scenarios."""
    
    processor = UnifiedVLProcessor(model_size="7B")
    
    # Scenario 1: Scientific paper with graphs
    print("=== Scenario 1: Scientific Paper ===")
    result = processor.smart_routing(
        "research_paper.png",
        "What was the improvement in accuracy?"
    )
    print(f"Routing: {result['routing']}")
    print(f"Reasoning: {result['reasoning']}")
    
    # # Scenario 2: Math problem with diagram
    # print("\n=== Scenario 2: Geometry Problem ===")
    # result = processor.smart_routing(
    #     "geometry_problem.png",
    #     "Find the area of the shaded region"
    # )
    # print(f"Routing: {result['routing']}")
    # print(f"Visual dependency: {result['visual_dependency']}")
    
    # # Scenario 3: Text document
    # print("\n=== Scenario 3: Text Document ===")
    # result = processor.smart_routing(
    #     "contract.png",
    #     "What is the termination clause?"
    # )
    # print(f"Routing: {result['routing']}")
    # print(f"Relevant content: {result['relevant_content'][:100]}...")
    
    # # Scenario 4: Data table
    # print("\n=== Scenario 4: Data Table ===")
    # result = processor.smart_routing(
    #     "sales_table.png",
    #     "What was the total revenue in Q3?"
    # )
    print(f"Routing: {result['routing']}")
    print(f"Confidence: {result['confidence']}")


# Batch processing with intelligent routing
class BatchDocumentProcessor:
    """Process multiple documents with intelligent routing."""
    
    def __init__(self, processor: UnifiedVLProcessor):
        self.processor = processor
        self.text_model_queue = []
        self.vision_model_queue = []
    
    def process_batch(self, documents: list[tuple[str, str]]) -> dict:
        """
        Process a batch of (image_path, question) tuples.
        
        Returns:
            Dictionary with routing statistics and queues
        """
        
        for image_path, question in documents:
            result = self.processor.smart_routing(image_path, question)
            
            if result["routing"] == "text_model":
                self.text_model_queue.append({
                    "image": image_path,
                    "question": question,
                    "relevant_content": result["relevant_content"],
                    "confidence": result["confidence"]
                })
            else:
                self.vision_model_queue.append({
                    "image": image_path,
                    "question": question,
                    "visual_dependency": result["visual_dependency"],
                    "confidence": result["confidence"]
                })
        
        return {
            "total_documents": len(documents),
            "text_model_count": len(self.text_model_queue),
            "vision_model_count": len(self.vision_model_queue),
            "text_model_queue": self.text_model_queue,
            "vision_model_queue": self.vision_model_queue
        }
    
    def get_routing_stats(self) -> dict:
        """Get statistics about routing decisions."""
        
        text_confidences = [item["confidence"] for item in self.text_model_queue]
        vision_confidences = [item["confidence"] for item in self.vision_model_queue]
        
        return {
            "text_model": {
                "count": len(self.text_model_queue),
                "avg_confidence": sum(text_confidences) / len(text_confidences) if text_confidences else 0
            },
            "vision_model": {
                "count": len(self.vision_model_queue),
                "avg_confidence": sum(vision_confidences) / len(vision_confidences) if vision_confidences else 0
            }
        }


# Advanced prompt engineering for better classification
class AdvancedClassifier(UnifiedVLProcessor):
    """Advanced classifier with specialized prompts for different document types."""
    
    def classify_with_examples(self, image_path: str, user_question: str = "") -> dict:
        """Classify using few-shot examples for better accuracy."""
        
        prompt = f"""You are an expert at analyzing documents and determining whether visual information is necessary to answer questions.

Examples of TEXT-ONLY questions:
1. "What is the definition of X?" - Can be answered from text
2. "Summarize the main points" - Text content is sufficient
3. "What date was mentioned?" - Text extraction is enough
4. "What are the terms and conditions?" - Legal text doesn't need visuals

Examples of VISUAL-NEEDED questions:
1. "Which bar is highest in the chart?" - Requires seeing the visual
2. "What is the angle in the diagram?" - Needs geometric visualization
3. "What does the flowchart show?" - Requires understanding visual flow
4. "What trend does the graph display?" - Needs to see data visualization

Now analyze this document:
User's question: "{user_question if user_question else 'General analysis'}"

Determine:
1. Can this question be answered using only text content?
2. Are visual elements (charts, diagrams, spatial layouts) necessary?

Respond with:
CLASSIFICATION: text_only OR needs_visual
KEY_FACTORS: [list the key factors in your decision]
CONFIDENCE: [0.0-1.0]"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process and generate response
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]
        
        return self._parse_classification(output, user_question)
    
    def _parse_classification(self, output: str, user_question: str) -> dict:
        """Parse classification output."""
        
        classification = "needs_visual"  # default
        confidence = 0.5
        key_factors = []
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("CLASSIFICATION:"):
                class_value = line[15:].strip().lower()
                classification = "text_only" if "text_only" in class_value else "needs_visual"
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[11:].strip())
                except:
                    pass
            elif line.startswith("KEY_FACTORS:"):
                factors = line[12:].strip()
                key_factors = [f.strip() for f in factors.split(',')]
        
        return {
            "question": user_question,
            "classification": classification,
            "routing": "text_model" if classification == "text_only" else "vision_model",
            "confidence": confidence,
            "key_factors": key_factors
        }


if __name__ == "__main__":
    # Run examples
    example_scenarios()
    
    # Demonstrate batch processing
    processor = UnifiedVLProcessor()
    batch_processor = BatchDocumentProcessor(processor)
    
    # Example batch
    documents = [
        ("image.png", "Solve this question for me."),
        # ("chart1.png", "What is the trend?"),
        # ("table1.png", "What is the total?"),
        # ("diagram1.png", "How does the process work?")
    ]
    
    results = batch_processor.process_batch(documents)
    print(f"\nBatch Processing Results:")
    print(json.dumps(batch_processor.get_routing_stats(), indent=2))