import os
import numpy as np
from typing import List, Dict
from pathlib import Path
import pickle
from docling.document_converter import PdfFormatOption
# Docling imports - updated for latest API
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# Embedding and vector search imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For text splitting
import re


class DoclingRAG:
    """RAG system using Docling for PDF extraction and semantic search"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the RAG system
        
        Args:
            embedding_model: Name of the sentence transformer model
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.chunk_metadata = []
        
        # Initialize Docling converter with PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Set to True if you need OCR
        pipeline_options.do_table_structure = True  # Extract tables
        
        self.converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using Docling
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        print(f"Extracting text from: {pdf_path}")
        
        try:
            # Convert PDF using Docling
            conversion_result = self.converter.convert(pdf_path)
            
            # The conversion result should have a 'document' attribute
            document = conversion_result.document
            
            # Export to markdown format (preserves structure better)
            full_text = document.export_to_markdown()
            
            # Alternative: Export to plain text if markdown doesn't work
            if not full_text:
                full_text = document.export_to_text()
            
            print(f"Extracted {len(full_text)} characters from {os.path.basename(pdf_path)}")
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            print(f"Trying alternative extraction method...")
            
            # Alternative method: Try to access the document content directly
            try:
                conversion_result = self.converter.convert(pdf_path)
                
                # Try to iterate through the document's content
                full_text = ""
                
                # Check if we can iterate through pages or sections
                if hasattr(conversion_result, 'pages'):
                    for page in conversion_result.pages:
                        if hasattr(page, 'text'):
                            full_text += page.text + "\n"
                
                # If still no text, try string representation
                if not full_text:
                    full_text = str(conversion_result.document)
                    
            except Exception as e2:
                print(f"Alternative extraction also failed: {e2}")
                return ""
        
        return full_text.strip()
    
    def split_text_into_chunks(self, text: str, source: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            source: Source document name
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text)
        
        # Split into sentences for better chunk boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': source,
                    'chunk_id': len(chunks)
                })
                
                # Create overlap by keeping last part of current chunk
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                current_chunk += " " + sentence
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'source': source,
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def add_pdf_to_index(self, pdf_path: str):
        """
        Add a PDF to the RAG index
        
        Args:
            pdf_path: Path to the PDF file
        """
        # Extract text using Docling
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print(f"Warning: No text extracted from {pdf_path}")
            return
        
        # Store the full document
        self.documents.append({
            'path': pdf_path,
            'text': text,
            'name': os.path.basename(pdf_path)
        })
        
        # Split into chunks
        new_chunks = self.split_text_into_chunks(text, os.path.basename(pdf_path))
        self.chunks.extend(new_chunks)
        
        print(f"Added {len(new_chunks)} chunks from {os.path.basename(pdf_path)}")
    
    def build_embeddings(self):
        """Build embeddings for all chunks"""
        if not self.chunks:
            raise ValueError("No chunks to embed. Add PDFs first.")
        
        print(f"Building embeddings for {len(self.chunks)} chunks...")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Create embeddings
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        print("Embeddings built successfully!")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks based on query
        
        Args:
            query: User question
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with scores
        """
        if self.embeddings is None:
            raise ValueError("No embeddings built. Run build_embeddings() first.")
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx]['text'],
                'source': self.chunks[idx]['source'],
                'score': float(similarities[idx]),
                'chunk_id': self.chunks[idx]['chunk_id']
            })
        
        return results
    
    def save_index(self, path: str):
        """Save the RAG index to disk"""
        data = {
            'documents': self.documents,
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'chunk_metadata': self.chunk_metadata
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load the RAG index from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        self.chunk_metadata = data.get('chunk_metadata', [])
        
        print(f"Index loaded from {path}")


# Alternative simple extractor if Docling has issues
def extract_with_fallback(pdf_path: str) -> str:
    """
    Extract text with fallback to PyPDF2 if Docling fails
    """
    try:
        # Try Docling first
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        # Try to get markdown export
        if hasattr(result.document, 'export_to_markdown'):
            return result.document.export_to_markdown()
        # Try plain text export
        elif hasattr(result.document, 'export_to_text'):
            return result.document.export_to_text()
        else:
            raise Exception("Docling export methods not found")
            
    except Exception as e:
        print(f"Docling failed: {e}. Trying PyPDF2...")
        
        try:
            import PyPDF2
            
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            return text
            
        except ImportError:
            print("PyPDF2 not installed. Install it with: pip install PyPDF2")
            return ""
        except Exception as e:
            print(f"PyPDF2 also failed: {e}")
            return ""


def test_docling_api():
    """Test function to understand Docling API structure"""
    from docling.document_converter import DocumentConverter
    
    # Test with a single PDF
    pdf_path = "test.pdf"  # Replace with your PDF path
    
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    
    print("=== Docling API Inspection ===")
    print(f"Result type: {type(result)}")
    print(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
    
    if hasattr(result, 'document'):
        doc = result.document
        print(f"\nDocument type: {type(doc)}")
        print(f"Document attributes: {[attr for attr in dir(doc) if not attr.startswith('_')]}")
        
        # Try different export methods
        print("\n=== Testing export methods ===")
        
        if hasattr(doc, 'export_to_markdown'):
            try:
                md_text = doc.export_to_markdown()
                print(f"Markdown export successful: {len(md_text)} chars")
                print(f"First 200 chars: {md_text[:200]}")
            except Exception as e:
                print(f"Markdown export failed: {e}")
        
        if hasattr(doc, 'export_to_text'):
            try:
                text = doc.export_to_text()
                print(f"Text export successful: {len(text)} chars")
                print(f"First 200 chars: {text[:200]}")
            except Exception as e:
                print(f"Text export failed: {e}")


def main():
    """Example usage of the DoclingRAG system"""
    
    # Initialize RAG system
    rag = DoclingRAG(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Add PDFs to the index
    pdf_folder = "/home/mukesh/extramarks/rag/books"  # Your PDF folder
    
    # Process PDFs
    if os.path.exists(pdf_folder):
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            try:
                rag.add_pdf_to_index(str(pdf_file))
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                
                # Try fallback method
                print("Trying fallback extraction...")
                text = extract_with_fallback(str(pdf_file))
                if text:
                    # Manually add to documents and chunks
                    rag.documents.append({
                        'path': str(pdf_file),
                        'text': text,
                        'name': pdf_file.name
                    })
                    chunks = rag.split_text_into_chunks(text, pdf_file.name)
                    rag.chunks.extend(chunks)
                    print(f"Fallback successful: Added {len(chunks)} chunks")
    
    if rag.chunks:
        # Build embeddings
        rag.build_embeddings()
        
        # Save the index
        rag.save_index("rag_index.pkl")
        
        # Test search
        test_query = "What did Liberal Nationalism Stand for?"
        print(f"\nTest search for: '{test_query}'")
        results = rag.search(test_query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.3f})")
            print(f"Source: {result['source']}")
            print(f"Text: {result['text']}...")
    else:
        print("No chunks created. Please check your PDFs.")


if __name__ == "__main__":
    # Uncomment to test Docling API
    # test_docling_api()
    
    # Run main
    main()