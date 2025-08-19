import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import base64
import io
import re
import os
from typing import Dict, List, Tuple, Optional, Union
import logging
from openai import OpenAI  # Requires: pip install openai>=1.0

# RAG imports
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSearcher:
    """RAG search functionality for SST subjects."""
    
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG searcher.
        
        Args:
            persist_directory: Path to the persisted Chroma database
            embedding_model: HuggingFace embedding model name
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized embeddings with {embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
    
    def load_vectorstore(self) -> bool:
        """
        Load existing vector store from persist directory.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.persist_directory) and self.embeddings is not None:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="docling_rag"
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}  # Return top 3 most relevant documents
                )
                
                # Get collection info
                collection = self.vectorstore._collection
                count = collection.count()
                logger.info(f"✓ Loaded existing RAG vector store with {count} documents")
                return True
            else:
                logger.warning(f"RAG vector store directory {self.persist_directory} not found or embeddings not initialized")
                return False
        except Exception as e:
            logger.error(f"Could not load RAG vector store: {e}")
            return False
    
    def search_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant context in the vector store.
        
        Args:
            query: Search query (the question)
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if self.vectorstore is None:
            logger.warning("Vector store not loaded. Cannot perform RAG search.")
            return []
        
        try:
            # Perform similarity search with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    'text': doc.page_content,
                    'source': doc.metadata.get('source_file', 'Unknown'),
                    'page_number': doc.metadata.get('page_number', 'N/A'),
                    'score': float(1 - score),  # Convert distance to similarity score
                    'metadata': doc.metadata
                })
            
            logger.info(f"Found {len(results)} relevant documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error during RAG search: {e}")
            return []
    
    def format_context_for_llm(self, search_results: List[Dict]) -> str:
        """
        Format search results into context string for LLM.
        
        Args:
            search_results: List of search results from RAG
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant context found in the knowledge base."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result.get('source', 'Unknown')
            page = result.get('page_number', 'N/A')
            text = result.get('text', '').strip()
            score = result.get('score', 0)
            
            context_part = f"""
Context {i} (Source: {source}, Page: {page}, Relevance: {score:.2f}):
{text}
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)


class QuestionSolvingPipeline:
 
    def __init__(self, openai_api_key: str, gpt_model: str = "o3", rag_persist_directory: str = "./chroma_db"):
    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        
        # Set up OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        # GPT model configuration (default to O3)
        self.gpt_model = gpt_model
        
        # Initialize RAG searcher for SST
        self.rag_searcher = RAGSearcher(persist_directory=rag_persist_directory)
        self.rag_enabled = self.rag_searcher.load_vectorstore()
        
        if self.rag_enabled:
            logger.info("RAG system loaded successfully for SST questions")
        else:
            logger.warning("RAG system not available - SST questions will be solved without context")
        
        # Load base vision model (Qwen2.5-VL)
        self.load_base_model()
        
        # Model configurations for different specializations
        self.model_configs = {
            "math": "Qwen/Qwen2.5-Math-72B-Instruct",
            "physics": "Qwen/Qwen2.5-Math-72B-Instruct",  # Using math model for physics too
            "biology": "google/medgemma-27b-it",
            "chemistry": "google/medgemma-27b-it",  # Using medgemma for chemistry too
            "sst": "meta-llama/Llama-3.1-8B-Instruct"  # Using general model for SST
        }
        
        # MedGemma model and processor (loaded separately for vision capabilities)
        self.medgemma_model = None
        self.medgemma_processor = None
        
        # Difficulty levels
        self.difficulty_levels = {
            1: "basic",
            2: "elementary", 
            3: "moderate",
            4: "advanced",
            5: "competitive"
        }
        
        # Subject categories
        self.subjects = ["math", "physics", "biology", "chemistry", "sst"]
        
    def load_base_model(self):
        """Load the base Qwen2.5-VL vision-language model."""
        try:
            model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            self.base_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.base_model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Base Qwen2.5-VL model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
    
    def load_medgemma_model(self):
        """Load MedGemma 27B model for biology and chemistry questions."""
        if self.medgemma_model is not None:
            return self.medgemma_model, self.medgemma_processor
        
        try:
            model_id = "google/medgemma-27b-it"
            self.medgemma_model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.medgemma_processor = AutoProcessor.from_pretrained(model_id)
            
            logger.info("MedGemma 27B model loaded successfully")
            return self.medgemma_model, self.medgemma_processor
            
        except Exception as e:
            logger.error(f"Error loading MedGemma model: {e}")
            return None, None
    
    def load_specialized_model(self, model_type: str):
        """Load specialized models on demand."""
        if model_type in self.models:
            return self.models[model_type], self.tokenizers[model_type]
        
        try:
            model_name = self.model_configs[model_type]
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
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {e}")
            return None, None
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def load_image_for_qwen(self, image_input: Union[str, None]) -> Optional[Image.Image]:
        """
        Load image for Qwen model processing.
        
        Args:
            image_input: Path to image file or base64 encoded string or None
            
        Returns:
            PIL Image object or None
        """
        if image_input is None:
            return None
            
        try:
            if os.path.exists(image_input):
                # It's a file path
                return Image.open(image_input).convert('RGB')
            else:
                # Assume it's base64 encoded
                if image_input.startswith('data:'):
                    # Remove data URL prefix
                    image_input = image_input.split(',')[1]
                image_bytes = base64.b64decode(image_input)
                return Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def to_data_url(self, image_data: str) -> str:
        """Convert base64 image data to data URL format."""
        # If it's already a data URL, return as is
        if image_data.startswith('data:'):
            return image_data
        # Otherwise, create a data URL
        return f"data:image/jpeg;base64,{image_data}"
    
    def has_image(self, question_data: Dict) -> bool:
        """Check if the question contains an image."""
        return "image" in question_data and question_data["image"] is not None
    
    def classify_difficulty_and_subject(self, question: str, image_input: Optional[str] = None) -> Tuple[int, str]:
        """Classify question difficulty (1-5) and subject using Qwen2.5-VL model with both text and image."""
        
        classification_prompt = f"""
You are an expert question classifier for academic questions. Analyze the following question and any provided image to determine:

1. Difficulty level (1-5):
   **Level 1 – Basic**
    Questions involve direct, single-step calculations or recall of basic facts. No reasoning required.

    **Level 2 – Elementary**
    Slightly more involved than Level 1. May require interpreting short text, combining 2 steps, or applying elementary concepts.

    **Level 3 – Moderate**
    Requires understanding of concepts and multiple steps. Involves basic algebra, geometry, or standard textbook problems.

    **Level 4 – Advanced**
    Complex problems requiring multi-step logic, advanced concepts, or integration of multiple topics. Typical of competitive exams.

    **Level 5 – Highly Competitive**
    Extremely challenging problems requiring deep understanding, creativity, and mastery of multiple concepts. JEE Advanced/Olympiad level.

2. Subject category (choose ONLY from these):
   - math: Mathematics (arithmetic, algebra, calculus, geometry, trigonometry, statistics, probability)
   - physics: Physics (mechanics, electricity, magnetism, thermodynamics, optics, waves, modern physics)
   - biology: Biology (cells, genetics, anatomy, physiology, ecology, evolution, botany, zoology)
   - chemistry: Chemistry (reactions, periodic table, organic/inorganic chemistry, physical chemistry, stoichiometry)
   - sst: Social Studies (history, geography, civics, economics, political science, sociology)

Classification Rules:
- Analyze both the question text and any provided image
- If the image shows mathematical equations, graphs, geometric figures → "math"
- If the image shows physics diagrams, circuits, forces, experimental setups → "physics"
- If the image shows biological structures, organisms, cells, anatomical diagrams → "biology"
- If the image shows chemical equations, molecular structures, lab equipment → "chemistry"
- If the image shows maps, historical documents, social contexts → "sst"
- If the question text involves calculations, equations, mathematical concepts → "math"
- If it involves physical phenomena, forces, energy, motion → "physics"
- If it involves living organisms, life processes → "biology"
- If it involves chemical substances, reactions, elements → "chemistry"
- If it involves society, history, geography, government, economics → "sst"

Question: {question}

Provide your classification in this exact format:
Difficulty: [1-5]
Subject: [math/physics/biology/chemistry/sst]
"""

        try:
            # Load image if provided
            image = self.load_image_for_qwen(image_input)
            
            # Prepare messages for Qwen2.5-VL
            if image is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": classification_prompt}
                        ]
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": classification_prompt}]
                    }
                ]
            
            # Apply chat template
            text = self.base_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            if image is not None:
                inputs = self.base_processor(
                    text=text, images=image, return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.base_processor(
                    text=text, return_tensors="pt"
                ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.base_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.base_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Parse response
            difficulty_match = re.search(r"Difficulty:\s*(\d+)", response)
            subject_match = re.search(r"Subject:\s*(\w+)", response)
            
            difficulty = int(difficulty_match.group(1)) if difficulty_match else 3
            subject = subject_match.group(1).lower() if subject_match else "math"
            
            # Validate ranges
            difficulty = max(1, min(5, difficulty))
            if subject not in self.subjects:
                subject = "math"  # Default to math if invalid
            
            logger.info(f"Classified - Difficulty: {difficulty}, Subject: {subject}")
            return difficulty, subject
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return 3, "math"  # Default fallback
    
    def solve_with_base_model(self, question: str, image_input: Optional[str] = None) -> str:
        """Solve easy questions (Level 1-2) with Qwen2.5-VL model."""
        solve_prompt = f"""
You are a helpful AI assistant. Solve the following question step by step.
Analyze any provided image carefully and provide a clear, detailed explanation of your solution.

Question: {question}

Solution:
"""
        
        try:
            # Load image if provided
            image = self.load_image_for_qwen(image_input)
            
            # Prepare messages
            if image is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": solve_prompt}
                        ]
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": solve_prompt}]
                    }
                ]
            
            # Apply chat template
            text = self.base_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            if image is not None:
                inputs = self.base_processor(
                    text=text, images=image, return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.base_processor(
                    text=text, return_tensors="pt"
                ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.base_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            solution = self.base_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return solution
            
        except Exception as e:
            logger.error(f"Error solving with base model: {e}")
            return "Sorry, I couldn't solve this question."
    
    def solve_with_gpt(self, question: str, subject: str, image_data: Optional[str] = None, context: Optional[str] = None) -> str:
        """Solve Level 4-5 questions using GPT O3 (both text and image), with optional RAG context."""
        try:
            # Prepare prompts based on subject
            subject_prompts = {
                "math": "You are an expert mathematician. Solve this mathematical problem with detailed step-by-step solutions: ",
                "physics": "You are an expert physicist. Solve this physics problem with clear explanations and proper formulas: ",
                "biology": "You are an expert biologist. Answer this biology question with detailed scientific explanations: ",
                "chemistry": "You are an expert chemist. Solve this chemistry problem with proper equations and explanations: ",
                "sst": "You are an expert in social studies. Answer this question about history, geography, civics, or economics comprehensively: "
            }
            
            # Build the prompt with context if available
            base_prompt = subject_prompts.get(subject, subject_prompts["math"])
            
            if context and subject == "sst":
                full_prompt = f"""{base_prompt}

Use the following context from relevant textbooks and educational materials to help answer the question:

CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive answer using the context provided above along with your knowledge. If the context contains relevant information, make sure to incorporate it into your response."""
            else:
                full_prompt = base_prompt + question
            
            # Build message content
            if image_data:
                # Check if image_data is a file path or base64 string
                if os.path.exists(image_data):
                    # It's a file path, encode it
                    base64_image = self.encode_image(image_data)
                else:
                    # Assume it's already base64 encoded
                    base64_image = image_data
                
                # Create data URL
                image_url = f"data:image/jpeg;base64,{base64_image}"
                
                # Use the exact format from the reference
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ],
                    }
                ]
            else:
                # Text-only message
                messages = [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ]
            
            # Call GPT O3 API using the exact format from example
            response = self.client.chat.completions.create(
                model=self.gpt_model,  # Uses "o3" by default
                messages=messages,
            )
            
            solution = response.choices[0].message.content
            return solution
            
        except Exception as e:
            logger.error(f"Error with GPT API using model {self.gpt_model}: {e}")
            return f"Error using GPT {self.gpt_model}: {str(e)}"
    
    def solve_with_medgemma(self, question: str, image_input: Optional[str] = None) -> str:
        """Solve biology/chemistry questions with MedGemma 27B model (supports both text and images)."""
        model, processor = self.load_medgemma_model()
        
        if model is None or processor is None:
            return "Sorry, the MedGemma model is not available."
        
        try:
            # Load image if provided
            image = self.load_image_for_qwen(image_input)
            
            # Create messages for MedGemma
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful medical and scientific assistant specializing in biology and chemistry."}]
                }
            ]
            
            # Build user message content
            user_content = [{"type": "text", "text": f"Answer this question with detailed scientific explanation: {question}"}]
            
            # Add image if available
            if image is not None:
                user_content.append({"type": "image", "image": image})
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Process inputs using MedGemma's processor
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response
            with torch.inference_mode():
                generation = model.generate(
                    **inputs, 
                    max_new_tokens=1024, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                generation = generation[0][input_len:]
            
            solution = processor.decode(generation, skip_special_tokens=True)
            return solution
            
        except Exception as e:
            logger.error(f"Error solving with MedGemma model: {e}")
            return "Sorry, I couldn't solve this question with the MedGemma model."
    
    def solve_with_specialized_model(self, question: str, subject: str, context: Optional[str] = None) -> str:
        """Solve Level 3 questions with specialized open-source models, with optional RAG context for SST."""
        model, tokenizer = self.load_specialized_model(subject)
        
        if model is None or tokenizer is None:
            return "Sorry, the specialized model is not available."
        
        # Create appropriate prompt based on subject, with context for SST
        if subject == "sst" and context:
            prompt = f"""You are an expert in social studies. Use the following context from relevant textbooks and educational materials to help answer the question comprehensively.

    CONTEXT:
    {context}

    QUESTION: {question}

    Provide a comprehensive answer using the context provided above along with your knowledge. If the context contains relevant information, make sure to incorporate it into your response.

    Answer:"""
        else:
            # Standard prompts for other subjects
            prompts = {
                "math": f"Solve this mathematics problem step by step:\n\n{question}\n\nSolution:",
                "physics": f"Solve this physics problem with proper formulas and explanations:\n\n{question}\n\nSolution:",
                "sst": f"Answer this social studies question comprehensively:\n\n{question}\n\nAnswer:"
            }
            prompt = prompts.get(subject, prompts["math"])
        
        try:
            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get the length of input tokens to extract only the generated part
            input_length = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,  # Reduced from 1024 to prevent overly long responses
                    temperature=0.3,     # Slightly increased for better diversity
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Add repetition penalty to reduce repetitive text
                    top_p=0.9,              # Add nucleus sampling for better quality
                    early_stopping=True      # Stop generation early when appropriate
                )
            
            # Extract only the generated tokens (excluding the input prompt)
            generated_tokens = outputs[0][input_length:]
            solution = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up the response - remove any remaining artifacts
            solution = solution.strip()
            
            # Remove common unwanted patterns that might appear at the end
            unwanted_patterns = [
                r'\n\n.*Best regards.*',  # Remove signature-like patterns
                r'\n\n.*I hope this helps.*',  # Remove help messages
                r'\n\n.*Let me know.*',   # Remove follow-up requests
                r'\n\n.*Note:.*$',       # Remove trailing notes
            ]
            
            for pattern in unwanted_patterns:
                solution = re.sub(pattern, '', solution, flags=re.DOTALL)
            
            # Final cleanup - remove excessive newlines and whitespace
            solution = re.sub(r'\n{3,}', '\n\n', solution)  # Replace 3+ newlines with 2
            solution = solution.strip()
            
            # If the response is still too repetitive or contains unwanted content, truncate it
            lines = solution.split('\n')
            cleaned_lines = []
            seen_lines = set()
            
            for line in lines:
                line = line.strip()
                # Skip empty lines that are too frequent
                if line == '' and len(cleaned_lines) > 0 and cleaned_lines[-1] == '':
                    continue
                # Skip repetitive lines
                if line in seen_lines and len(line) > 10:  # Only check for repetition in longer lines
                    continue
                # Stop if we encounter signature-like patterns
                if any(phrase in line.lower() for phrase in ['best regards', 'let me know', 'i hope this helps']):
                    break
                
                cleaned_lines.append(line)
                if line:  # Only add non-empty lines to seen_lines
                    seen_lines.add(line)
            
            solution = '\n'.join(cleaned_lines).strip()
            
            # Final check - if response is empty or too short, provide a fallback
            if len(solution.strip()) < 20:
                return "I apologize, but I couldn't generate a proper response for this question."
            
            return solution
            
        except Exception as e:
            logger.error(f"Error solving with {subject} model: {e}")
            return "Sorry, I couldn't solve this question with the specialized model."
    
    def solve_question(self, question_data: Dict) -> Dict:
        """Main pipeline function to solve questions with RAG integration for SST."""
        question = question_data.get("question", "")
        image_data = question_data.get("image", None)
        
        if not question:
            return {"error": "No question provided"}
        
        # Classify difficulty and subject using both text and image
        difficulty, subject = self.classify_difficulty_and_subject(question, image_data)
        
        # Initialize context variables
        context = None
        rag_search_results = []
        
        # RAG search for SST questions
        if subject == "sst" and self.rag_enabled:
            logger.info("Performing RAG search for SST question")
            rag_search_results = self.rag_searcher.search_context(question, top_k=3)
            if rag_search_results:
                context = self.rag_searcher.format_context_for_llm(rag_search_results)
                logger.info(f"Found {len(rag_search_results)} relevant context documents")
            else:
                logger.warning("No relevant context found in RAG search")
        
        # Special routing for biology and chemistry - always use MedGemma regardless of difficulty
        if subject in ["biology", "chemistry"]:
            logger.info(f"Solving {subject} question with MedGemma 27B model (Level {difficulty})")
            solution = self.solve_with_medgemma(question, image_data)
            model_used = "medgemma_27b_model"
        
        # Special routing for SST - always use Llama-3.1-8B-Instruct regardless of difficulty, WITH RAG context
        elif subject == "sst":
            logger.info(f"Solving SST question with Llama-3.1-8B-Instruct model (Level {difficulty}) {'with RAG context' if context else 'without context'}")
            solution = self.solve_with_specialized_model(question, subject, context)
            model_used = "llama_3.1_8b_instruct_with_rag" if context else "llama_3.1_8b_instruct"
        
        # Route other subjects based on difficulty
        elif difficulty <= 2:
            # Easy questions - use base Qwen2.5-VL model
            logger.info(f"Solving Level {difficulty} question with Qwen2.5-VL base model")
            solution = self.solve_with_base_model(question, image_data)
            model_used = "qwen2.5_vl_base_model"
            
        elif difficulty == 3:
            # Moderate questions - use specialized open-source models
            logger.info(f"Solving Level {difficulty} {subject} question with specialized model")
            solution = self.solve_with_specialized_model(question, subject)
            model_used = f"specialized_{subject}_model"
            
        else:  # difficulty >= 4
            # Advanced questions - use GPT O3 for both text and image, with context for SST
            logger.info(f"Solving Level {difficulty} {subject} question with {self.gpt_model} {'with RAG context' if context and subject == 'sst' else ''}")
            solution = self.solve_with_gpt(question, subject, image_data, context)
            model_used = f"{self.gpt_model}_with_rag" if context and subject == "sst" else self.gpt_model
        
        return {
            "question": question,
            "solution": solution,
            "difficulty": difficulty,
            "difficulty_name": self.difficulty_levels[difficulty],
            "subject": subject,
            "model_used": model_used,
            "has_image": image_data is not None,
            "rag_context_found": len(rag_search_results) > 0 if subject == "sst" else False,
            "rag_search_results": rag_search_results if subject == "sst" else []
        }


# Usage example and API wrapper
class QuestionSolvingAPI:
    def __init__(self, openai_api_key: str, gpt_model: str = "o3", rag_persist_directory: str = "./chroma_db"):
        self.pipeline = QuestionSolvingPipeline(openai_api_key, gpt_model, rag_persist_directory)
    
    def process_question(self, question: str, image: Optional[str] = None) -> Dict:
        """
        Process a question through the pipeline.
        
        Args:
            question: The question text
            image: Path to image file OR base64 encoded image string
        
        Returns:
            Dictionary with solution and metadata
        """
        question_data = {
            "question": question,
            "image": image
        }
        
        return self.pipeline.solve_question(question_data)


# Helper function to convert image file to base64
def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Example usage
if __name__ == "__main__":
    # Initialize the API with your OpenAI API key and RAG directory
    OPENAI_API_KEY = "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"
    RAG_DIRECTORY = "/home/karang/dikshant/extramarks_239/rag/chroma_db"  # Path to your RAG vector store
    
    # Using O3 model (default) for levels 4-5 with RAG integration
    api = QuestionSolvingAPI(OPENAI_API_KEY, rag_persist_directory=RAG_DIRECTORY)  # Defaults to "o3"
    
    # Or specify a different model if needed
    # api = QuestionSolvingAPI(OPENAI_API_KEY, "gpt-4", RAG_DIRECTORY)
    
    # Test cases
    test_cases = [
        # SST question (will use RAG)
        ('What did Liberal Nationalism Stand for?', None),
        ('Explain the concept of democracy and its importance in modern society', None),
        
        # Biology question (will use MedGemma)
        ('What is photosynthesis?', None),
        ('Which of the following are not the effects of parathyroid hormone?', '/home/karang/dikshant/extramarks_239/neet1.png'),
        
        # Math/Physics question (will use appropriate model based on difficulty)
        ('solve this question', '/home/karang/dikshant/extramarks_239/q4.jpg'),
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        
        if isinstance(test_case, tuple) and len(test_case) == 2:
            question_text, image_path = test_case
            print(f"Question: {question_text}")
            if image_path:
                print(f"Image: {image_path}")
            print(f"{'='*80}")
            
            result = api.process_question(question_text, image=image_path)
            
            print(f"Difficulty: Level {result['difficulty']} ({result['difficulty_name']})")
            print(f"Subject: {result['subject']}")
            print(f"Model Used: {result['model_used']}")
            print(f"Has Image: {result['has_image']}")
            
            # Show RAG information for SST questions
            if result['subject'] == 'sst':
                print(f"RAG Context Found: {result['rag_context_found']}")
                if result['rag_context_found']:
                    print(f"RAG Sources: {len(result['rag_search_results'])} documents")
                    for j, rag_result in enumerate(result['rag_search_results'], 1):
                        print(f"  - Source {j}: {rag_result['source']} (Score: {rag_result['score']:.2f})")
            
            print(f"\nSolution:\n{result['solution']}")
            
            # Add a separator between test cases
            if i < len(test_cases):
                print(f"\n{'-'*80}")