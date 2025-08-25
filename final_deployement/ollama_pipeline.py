#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Question Solving Pipeline — same routing/logic as your original:
- Level 1–2: Open-source VL base (Ollama)
- Level 3: Open-source specialized models (Ollama)
- Level >= 4: Closed-source (OpenAI o3), with RAG for SST when available
- Biology/Chemistry: always the (open-source) bio/chem path as in your logic
- SST: always Llama-3.1-8B with RAG (open-source via Ollama), like your logic

Changes:
- All open-source models now go through a local Ollama server (HTTP).
- Removed direct Hugging Face model loads (no AutoModel/AutoTokenizer).
- Kept your function names and routing semantics (e.g., solve_with_medgemma still exists but calls Ollama).

Prereqs:
  pip install requests pillow langchain-core langchain-chroma langchain-huggingface openai

Env (optional):
  export OLLAMA_BASE_URL="http://localhost:11434"
  export VL_BASE_MODEL="qwen2.5vl:7b"         # Vision-language
  export MATH_MODEL="qwen2-math:72b"          # Math/Physics specialist
  export SST_MODEL="llama3.1:8b"              # SST/general
  export BIO_CHEM_MODEL="gemma3:27b"          # Biology/Chemistry text
  export OPENAI_API_KEY="sk-..."
"""

import os
import io
import re
import json
import base64
import logging
from typing import Dict, List, Tuple, Optional, Union

import requests
from PIL import Image  # kept in case you need simple image checks

# OpenAI (closed-source path for difficulty >= 4)
from openai import OpenAI

# RAG imports
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Ollama client
# -----------------------------------------------------------------------------
class OllamaClient:
    """
    Minimal Ollama REST client:
      - ensure_model(): pulls the model if missing
      - generate(): text or vision (images=list of base64 strings or file paths)
    """

    def __init__(self, base_url: Optional[str] = None, keep_alive: str = "30m", timeout: int = 300):
        self.base = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.keep_alive = keep_alive
        self.timeout = timeout

    def ensure_model(self, model: Optional[str]) -> None:
        if not model:
            return
        try:
            r = requests.post(
                f"{self.base}/api/pull",
                json={"model": model, "stream": False},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                logger.info(f"Ollama: model ready -> {model}")
            else:
                logger.warning(f"Ollama pull returned {r.status_code}: {r.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama pull failed for {model}: {e}")

    @staticmethod
    def _normalize_images(images: Optional[List[Union[str, bytes]]]) -> Optional[List[str]]:
        if not images:
            return None
        out = []
        for img in images:
            if isinstance(img, bytes):
                out.append(base64.b64encode(img).decode("utf-8"))
            elif isinstance(img, str) and os.path.exists(img):
                with open(img, "rb") as f:
                    out.append(base64.b64encode(f.read()).decode("utf-8"))
            elif isinstance(img, str):
                # assume already base64 or data URL
                if img.startswith("data:"):
                    img = img.split(",", 1)[1]
                out.append(img)
        return out

    def generate(
        self,
        model: str,
        prompt: str,
        images: Optional[List[Union[str, bytes]]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: Optional[int] = 1024,
        format: Optional[str] = None,   # e.g., "json"
        system: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> str:
        self.ensure_model(model)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "temperature": temperature,
            "top_p": top_p,
        }
        if num_predict is not None:
            payload["num_predict"] = num_predict
        if format:
            payload["format"] = format
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        if images:
            payload["images"] = self._normalize_images(images)

        try:
            r = requests.post(f"{self.base}/api/generate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generate error ({model}): {e}")
            return "Error: could not contact local LLM."

# -----------------------------------------------------------------------------
# RAG
# -----------------------------------------------------------------------------
class RAGSearcher:
    """RAG search functionality for SST subjects."""

    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized embeddings with {embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")

    def load_vectorstore(self) -> bool:
        try:
            if os.path.exists(self.persist_directory) and self.embeddings is not None:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="docling_rag"
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                count = self.vectorstore._collection.count()
                logger.info(f"✓ Loaded existing RAG vector store with {count} documents")
                return True
            else:
                logger.warning(f"RAG vector store directory {self.persist_directory} not found or embeddings not initialized")
                return False
        except Exception as e:
            logger.error(f"Could not load RAG vector store: {e}")
            return False

    def search_context(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.vectorstore is None:
            logger.warning("Vector store not loaded. Cannot perform RAG search.")
            return []
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    'text': doc.page_content,
                    'source': doc.metadata.get('source_file', 'Unknown'),
                    'page_number': doc.metadata.get('page_number', 'N/A'),
                    'score': float(1 - score),
                    'metadata': doc.metadata
                })
            logger.info(f"Found {len(results)} relevant documents for query: '{query[:50]}...'")
            return results
        except Exception as e:
            logger.error(f"Error during RAG search: {e}")
            return []

    def format_context_for_llm(self, search_results: List[Dict]) -> str:
        if not search_results:
            return "No relevant context found in the knowledge base."
        parts = []
        for i, result in enumerate(search_results, 1):
            parts.append(
                f"Context {i} (Source: {result.get('source','Unknown')}, Page: {result.get('page_number','N/A')}, Relevance: {result.get('score',0):.2f}):\n{result.get('text','').strip()}"
            )
        return "\n\n".join(parts)

# -----------------------------------------------------------------------------
# Main pipeline with your original routing semantics
# -----------------------------------------------------------------------------
class QuestionSolvingPipeline:

    def __init__(self, openai_api_key: str, gpt_model: str = "o3", rag_persist_directory: str = "./chroma_db"):
        # Clients
        self.ollama = OllamaClient(base_url=os.environ.get("OLLAMA_BASE_URL"))
        self.client = OpenAI(api_key=openai_api_key)

        # GPT model configuration (default to O3)
        self.gpt_model = gpt_model

        # RAG
        self.rag_searcher = RAGSearcher(persist_directory=rag_persist_directory)
        self.rag_enabled = self.rag_searcher.load_vectorstore()
        if self.rag_enabled:
            logger.info("RAG system loaded successfully for SST questions")
        else:
            logger.warning("RAG system not available - SST questions will be solved without context")

        # -----------------------------
        # Open-source models (via Ollama)
        # -----------------------------
        self.vl_model = os.environ.get("VL_BASE_MODEL", "qwen2.5vl:32b")       # Vision-language base (Qwen2.5-VL)
        self.math_model = os.environ.get("MATH_MODEL", "qwen2.5vl:72b")      # Math/Physicsialist
        self.sst_model  = os.environ.get("SST_MODEL",  "llama3.1:8b")         # SST/general
        self.bio_model  = os.environ.get("BIO_CHEM_MODEL", "gemma3:27b")      # Biology/Chemistry text
        self.physics_model  = "qwen2.5vl:72b"   # Biology/Chemistry text

        # Difficulty labels & subjects
        self.difficulty_levels = {
            1: "basic",
            2: "elementary",
            3: "moderate",
            4: "advanced",
            5: "competitive"
        }
        self.subjects = ["math", "physics", "biology", "chemistry", "sst"]

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def to_data_url(image_data: str) -> str:
        if image_data.startswith('data:'):
            return image_data
        return f"data:image/jpeg;base64,{image_data}"

    @staticmethod
    def _image_to_b64_list(image_input: Optional[str]) -> Optional[List[str]]:
        if image_input is None:
            return None
        if os.path.exists(image_input):
            with open(image_input, "rb") as f:
                return [base64.b64encode(f.read()).decode("utf-8")]
        # assume already base64 or data URL
        if image_input.startswith("data:"):
            image_input = image_input.split(",", 1)[1]
        return [image_input]

    # -----------------------------
    # Classifier (Qwen2.5-VL via Ollama)
    # -----------------------------
    def classify_difficulty_and_subject(self, question: str, image_input: Optional[str] = None) -> Tuple[int, str]:
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
            images = self._image_to_b64_list(image_input)
            if images:
                response = self.ollama.generate(
                    model=self.vl_model,
                    prompt=classification_prompt,
                    images=images,
                    temperature=0.1,
                    num_predict=128
                )
            else:
                response = self.ollama.generate(
                    model=self.sst_model,
                    prompt=classification_prompt,
                    temperature=0.1,
                    num_predict=128
                )

            difficulty_match = re.search(r"Difficulty:\s*(\d+)", response)
            subject_match = re.search(r"Subject:\s*([A-Za-z]+)", response)

            difficulty = int(difficulty_match.group(1)) if difficulty_match else 3
            subject = subject_match.group(1).lower() if subject_match else "math"

            difficulty = max(1, min(5, difficulty))
            if subject not in self.subjects:
                subject = "math"

            logger.info(f"Classified - Difficulty: {difficulty}, Subject: {subject}")
            return difficulty, subject

        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return 3, "math"

    # -----------------------------
    # Easy solve (Level 1–2) with VL base (Ollama)
    # -----------------------------
    def solve_with_base_model(self, question: str, image_input: Optional[str] = None) -> str:
        solve_prompt = f"""
You are a helpful AI assistant. Solve the following question step by step.
Analyze any provided image carefully and provide a clear, detailed explanation of your solution.

Question: {question}

Solution:
""".strip()

        try:
            images = self._image_to_b64_list(image_input)
            if images:
                return self.ollama.generate(
                    model=self.vl_model,
                    prompt=solve_prompt,
                    images=images,
                    temperature=0.7,
                    num_predict=1024
                )
            else:
                return self.ollama.generate(
                    model=self.sst_model,
                    prompt=solve_prompt,
                    temperature=0.7,
                    num_predict=1024
                )
        except Exception as e:
            logger.error(f"Error solving with base model: {e}")
            return "Sorry, I couldn't solve this question."

    # -----------------------------
    # Closed-source path (Level >= 4) — unchanged
    # -----------------------------
    def solve_with_gpt(self, question: str, subject: str, image_data: Optional[str] = None, context: Optional[str] = None) -> str:
        """Solve Level 4-5 questions using GPT O3 (both text and image), with optional RAG context."""
        try:
            subject_prompts = {
                "math": "You are an expert mathematician. Solve this mathematical problem with detailed step-by-step solutions: ",
                "physics": "You are an expert physicist. Solve this physics problem with clear explanations and proper formulas: ",
                "biology": "You are an expert biologist. Answer this biology question with detailed scientific explanations: ",
                "chemistry": "You are an expert chemist. Solve this chemistry problem with proper equations and explanations: ",
                "sst": "You are an expert in social studies. Answer this question about history, geography, civics, or economics comprehensively: "
            }
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

            if image_data:
                if os.path.exists(image_data):
                    base64_image = self.encode_image(image_data)
                else:
                    base64_image = image_data
                image_url = f"data:image/jpeg;base64,{base64_image}"
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
                messages = [{"role": "user", "content": full_prompt}]

            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=messages,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error with GPT API using model {self.gpt_model}: {e}")
            return f"Error using GPT {self.gpt_model}: {str(e)}"

    # -----------------------------
    # Bio/Chem path — keep function name, use Ollama
    # -----------------------------
    def solve_with_medgemma(self, question: str, image_input: Optional[str] = None) -> str:
        """
        In your logic bio/chem always use the dedicated model.
        Here:
          - If an image is provided, we use the VL model for multimodal reasoning.
          - Otherwise, we use a strong bio/chem text model (gemma3:27b by default).
        """
        images = self._image_to_b64_list(image_input)
        try:
            if images:
                prompt = f"You are a medical/scientific assistant. Use the image and text to answer precisely.\n\nQuestion: {question}\n\nAnswer:"
                return self.ollama.generate(model=self.vl_model, prompt=prompt, images=images, num_predict=1024)
            else:
                prompt = f"You are a medical/scientific assistant. Provide a detailed, accurate explanation.\n\nQuestion: {question}\n\nAnswer:"
                return self.ollama.generate(model=self.bio_model, prompt=prompt, num_predict=1024)
        except Exception as e:
            logger.error(f"Error solving with bio/chem model: {e}")
            return "Sorry, I couldn't solve this question with the bio/chem model."

    # -----------------------------
    # Level 3 specialized (open-source via Ollama)
    # -----------------------------


    def solve_with_specialized_model(
        self,
        question: str,
        subject: str,
        context: Optional[str] = None,
        image_input: Optional[str] = None
    ) -> str:
       
        try:
            images = self._image_to_b64_list(image_input)
            model = self.vl_model
            # If we have an image AND a VL model configured, prefer multimodal reasoning
            if images and getattr(self, "vl_model", None):
                if subject == "sst":
                    # SST with RAG + image
                    prompt = f"""You are an expert in social studies. Use BOTH the image and the context (if helpful) to answer comprehensively.

    CONTEXT (optional, may be empty):
    {context or ''}

    QUESTION:
    {question}

    Instructions:
    - First, interpret any relevant details from the image (maps/charts/figures/diagrams, etc.).
    - Then, combine with textual reasoning (and the context above if relevant) to produce a clear, final answer.
    Answer:"""
                elif subject == "math":
                    model = self.math_model
                    prompt = f"""You are a math expert. Use BOTH the image (diagrams/figures/equations) and the text to solve step by step, then give a concise final answer.

    Problem:
    {question}

    Solution:"""
                elif subject == "physics":
                    model = self.physics_model
                    prompt = f"""You are a physics expert. Use BOTH the image (setups/diagrams/graphs) and the text. Show formulas/units and give a clear final result.

    Problem:
    {question}

    Solution:"""
                else:  # default multimodal
                    prompt = f"""Use BOTH the image and the text to answer precisely.

    Question:
    {question}

    Answer:"""

                return self.ollama.generate(
                    model=model,
                    prompt=prompt,
                    images=images,
                    temperature=0.3,
                    top_p=0.9,
                    num_predict=1024
                )

            # Otherwise: text-only specialists (previous behavior)
            if subject == "sst" and context:
                prompt = f"""You are an expert in social studies. Use the following context from relevant textbooks and educational materials to help answer the question comprehensively.

    CONTEXT:
    {context}

    QUESTION: {question}

    Provide a comprehensive answer using the context provided above along with your knowledge.
    Answer:"""
                model = self.sst_model
            else:
                prompts = {
                    "math": f"Solve this mathematics problem step by step:\n\n{question}\n\nSolution:",
                    "physics": f"Solve this physics problem with proper formulas and explanations:\n\n{question}\n\nSolution:",
                    "sst": f"Answer this social studies question comprehensively:\n\n{question}\n\nAnswer:"
                }
                prompt = prompts.get(subject, prompts["math"])
                model = self.math_model if subject in ["math", "physics"] else self.sst_model

            return self.ollama.generate(model=model, prompt=prompt, temperature=0.3, top_p=0.9, num_predict=1024)

        except Exception as e:
            logger.error(f"Error solving with specialized {subject} model: {e}")
            return "Sorry, I couldn't solve this question with the specialized model."

    # -----------------------------
    # Orchestrator — unchanged logic
    # -----------------------------
    def solve_question(self, question_data: Dict) -> Dict:
        """Main pipeline function to solve questions with RAG integration for SST."""
        question = question_data.get("question", "")
        image_data = question_data.get("image", None)

        if not question:
            return {"error": "No question provided"}

        # Classify difficulty and subject (uses VL if image else text)
        difficulty, subject = self.classify_difficulty_and_subject(question, image_data)

        # RAG for SST
        context = None
        rag_search_results = []
        if subject == "sst" and self.rag_enabled:
            logger.info("Performing RAG search for SST question")
            rag_search_results = self.rag_searcher.search_context(question, top_k=3)
            if rag_search_results:
                context = self.rag_searcher.format_context_for_llm(rag_search_results)
                logger.info(f"Found {len(rag_search_results)} relevant context documents")
            else:
                logger.warning("No relevant context found in RAG search")

        # Biology/Chemistry — always dedicated model (as per your logic)
        if subject in ["biology", "chemistry"]:
            logger.info(f"Solving {subject} question with Bio/Chem model (Level {difficulty})")
            solution = self.solve_with_medgemma(question, image_data)
            model_used = "vl_base" if image_data else "bio_chem_text"

        # SST — always Llama-3.1-8B with RAG context when available (as per your logic)
        elif subject == "sst":
            logger.info(f"Solving SST question with Llama-3.1-8B (Level {difficulty}) {'with RAG context' if context else 'without context'}")
            solution = self.solve_with_specialized_model(question, subject, context)
            model_used = "llama3.1_8b_with_rag" if context else "llama3.1_8b"

        # Other subjects by difficulty
        elif difficulty <= 2:
            # Easy — base VL (same logic)
            logger.info(f"Solving Level {difficulty} question with VL base model")
            solution = self.solve_with_base_model(question, image_data)
            model_used = "vl_base"

        elif difficulty == 3:
            logger.info(f"Solving Level {difficulty} {subject} question with specialized model")
            # ⬇️ pass image_data too (context only matters for SST; others can pass None or context)
            solution = self.solve_with_specialized_model(question, subject, None , image_data )
            model_used = f"specialized_{subject}"


        else:  # difficulty >= 4
            # Advanced — closed-source GPT O3 (same logic)
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

# -----------------------------------------------------------------------------
# API wrapper
# -----------------------------------------------------------------------------
class QuestionSolvingAPI:
    def __init__(self, openai_api_key: str, gpt_model: str = "o3", rag_persist_directory: str = "./chroma_db"):
        self.pipeline = QuestionSolvingPipeline(openai_api_key, gpt_model, rag_persist_directory)

    def process_question(self, question: str, image: Optional[str] = None) -> Dict:
        question_data = {"question": question, "image": image}
        return self.pipeline.solve_question(question_data)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Prefer env var for secrets
    OPENAI_API_KEY = "Please ENter API key "
    RAG_DIRECTORY = "/home/karang/dikshant/extramarks_239/rag/chroma_db"

    api = QuestionSolvingAPI(OPENAI_API_KEY, gpt_model=os.environ.get("GPT_MODEL", "o3"), rag_persist_directory=RAG_DIRECTORY)

    test_cases = [
        # SST question (will use RAG)
        # ('What did Liberal Nationalism Stand for?', None),
        # ('Explain the concept of democracy and its importance in modern society', None),

        # # Biology/Chemistry (will use Bio/Chem path)
        # ('What is photosynthesis?', None),
        # ('Which of the following are not the effects of parathyroid hormone?', '/path/to/neet1.png'),

        # Math/Physics (will use appropriate model based on difficulty)
        ('Solve this question', '/home/karang/dikshant/extramarks_239/q2.jpg'),
    ]

    for i, (q, img) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        print(f"Question: {q}")
        if img:
            print(f"Image: {img}")
        print(f"{'-'*80}")

        result = api.process_question(q, image=img)

        print(f"Difficulty: Level {result['difficulty']} ({result['difficulty_name']})")
        print(f"Subject: {result['subject']}")
        print(f"Model Used: {result['model_used']}")
        print(f"Has Image: {result['has_image']}")
        if result['subject'] == 'sst':
            print(f"RAG Context Found: {result['rag_context_found']}")
            if result['rag_context_found']:
                print(f"RAG Sources: {len(result['rag_search_results'])} documents")
                for j, rag_result in enumerate(result['rag_search_results'], 1):
                    print(f"  - Source {j}: {rag_result['source']} (Score: {rag_result['score']:.2f})")

        print(f"\nSolution:\n{result['solution']}")
        if i < len(test_cases):
            print(f"\n{'-'*80}")
