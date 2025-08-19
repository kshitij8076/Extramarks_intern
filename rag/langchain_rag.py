import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from tempfile import mkdtemp

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Docling LangChain integration
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker

# Vector store (using Chroma as it's simpler than Milvus for local use)
from langchain_chroma import Chroma
# Alternative: from langchain_milvus import Milvus

# For filtering complex metadata
from langchain_community.vectorstores.utils import filter_complex_metadata

# Optional: For LLM integration
try:
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Install langchain-openai or langchain-huggingface for LLM integration")


class LangChainDoclingRAG:
    """RAG system using LangChain with Docling for PDF extraction and semantic search"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,  # Increased from 500 to reduce excessive splitting
        chunk_overlap: int = 100,  # Increased proportionally
        export_type: ExportType = ExportType.DOC_CHUNKS,
        persist_directory: Optional[str] = None,
        max_token_length: int = 450  # Safe limit for all-MiniLM-L6-v2 (512 max)
    ):
        """
        Initialize the RAG system
        
        Args:
            embedding_model: HuggingFace embedding model name
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            export_type: Docling export type (DOC_CHUNKS, MARKDOWN, or TEXT)
            persist_directory: Directory to persist vector store
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.export_type = export_type
        self.persist_directory = persist_directory or mkdtemp()
        self.max_token_length = max_token_length
        
        # Initialize components
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better similarity
        )
        self.vectorstore = None
        self.retriever = None
        self.documents = []
        
        # Text splitter for non-chunked exports
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        print(f"Initialized LangChain Docling RAG with {embedding_model}")
        print(f"Export type: {export_type}")
        print(f"Persist directory: {self.persist_directory}")

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters for English)"""
        return len(text) // 4
    
    def _truncate_long_documents(self, documents: List[Document], max_length: int = None) -> List[Document]:
        """
        More intelligent document splitting that respects token limits
        """
        if max_length is None:
            max_length = self.max_token_length * 4  # Convert tokens to approximate characters
            
        processed_docs = []
        split_count = 0
        
        for doc in documents:
            estimated_tokens = self._estimate_tokens(doc.page_content)
            
            # Only split if significantly over the limit
            if estimated_tokens > self.max_token_length:
                content = doc.page_content.strip()
                
                # Try to split on natural boundaries first
                sentences = content.split('. ')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip() + '. '
                    
                    # Check if adding this sentence would exceed limit
                    if self._estimate_tokens(current_chunk + sentence) > self.max_token_length:
                        if current_chunk:  # Save current chunk if not empty
                            chunks.append(current_chunk.strip())
                            # Start new chunk with overlap
                            overlap_sentences = current_chunk.split('. ')[-2:]
                            current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                        else:
                            # Single sentence too long, force split
                            current_chunk = sentence
                    else:
                        current_chunk += sentence
                
                # Add the last chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Create document objects for chunks
                for i, chunk_content in enumerate(chunks):
                    if len(chunk_content.strip()) > 20:  # Only keep meaningful chunks
                        chunk_doc = Document(
                            page_content=chunk_content,
                            metadata={**doc.metadata, 'chunk_index': i, 'is_split': True}
                        )
                        processed_docs.append(chunk_doc)
                
                if len(chunks) > 1:
                    split_count += 1
                    if split_count <= 5:  # Only show first 5 to reduce spam
                        print(f"  Split long document into {len(chunks)} chunks (tokens: {estimated_tokens}→{self.max_token_length})")
                    elif split_count == 6:
                        print(f"  ... (suppressing further split messages)")
            else:
                processed_docs.append(doc)
        
        if split_count > 0:
            print(f"Total documents split: {split_count}")
        return processed_docs

    def extract_documents_from_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        Extract documents from PDF files using Docling
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of LangChain Document objects
        """
        all_docs = []
        
        for pdf_path in pdf_paths:
            try:
                print(f"Processing: {os.path.basename(pdf_path)}")
                
                # Initialize Docling loader
                if self.export_type == ExportType.DOC_CHUNKS:
                    # Use HybridChunker for automatic chunking
                    loader = DoclingLoader(
                        file_path=[pdf_path],
                        export_type=self.export_type,
                        chunker=HybridChunker(tokenizer=self.embedding_model_name)
                    )
                else:
                    # Use without chunker for other export types
                    loader = DoclingLoader(
                        file_path=[pdf_path],
                        export_type=self.export_type
                    )
                
                # Load documents
                docs = loader.load()
                
                # Add source metadata and filter complex metadata
                for doc in docs:
                    # Add simple metadata
                    doc.metadata.update({
                        'source_file': os.path.basename(pdf_path),
                        'full_path': pdf_path
                    })
                    
                    # Extract useful info from complex metadata before filtering
                    if 'doc_items' in doc.metadata:
                        try:
                            doc_items = doc.metadata['doc_items']
                            if isinstance(doc_items, list) and len(doc_items) > 0:
                                first_item = doc_items[0]
                                if 'prov' in first_item and isinstance(first_item['prov'], list) and len(first_item['prov']) > 0:
                                    prov = first_item['prov'][0]
                                    if 'page_no' in prov:
                                        doc.metadata['page_number'] = prov['page_no']
                                    if 'bbox' in prov:
                                        bbox = prov['bbox']
                                        doc.metadata['bbox_left'] = bbox.get('l', 0)
                                        doc.metadata['bbox_top'] = bbox.get('t', 0)
                                        doc.metadata['bbox_right'] = bbox.get('r', 0)
                                        doc.metadata['bbox_bottom'] = bbox.get('b', 0)
                                if 'label' in first_item:
                                    doc.metadata['content_type'] = first_item['label']
                                if 'content_layer' in first_item:
                                    doc.metadata['content_layer'] = first_item['content_layer']
                        except Exception as e:
                            print(f"Warning: Could not extract metadata from doc_items: {e}")
                    
                    # Extract origin info
                    if 'origin' in doc.metadata and isinstance(doc.metadata['origin'], dict):
                        origin = doc.metadata['origin']
                        if 'filename' in origin:
                            doc.metadata['original_filename'] = origin['filename']
                        if 'mimetype' in origin:
                            doc.metadata['mimetype'] = origin['mimetype']
                
                # Handle different export types
                if self.export_type == ExportType.DOC_CHUNKS:
                    # Documents are already chunked
                    processed_docs = docs
                elif self.export_type == ExportType.MARKDOWN:
                    # Split markdown by headers
                    from langchain_text_splitters import MarkdownHeaderTextSplitter
                    header_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=[
                            ("#", "Header_1"),
                            ("##", "Header_2"), 
                            ("###", "Header_3"),
                        ]
                    )
                    processed_docs = []
                    for doc in docs:
                        splits = header_splitter.split_text(doc.page_content)
                        for split in splits:
                            # Create new document with inherited metadata
                            new_doc = Document(
                                page_content=split.page_content,
                                metadata={**doc.metadata, **split.metadata}
                            )
                            processed_docs.append(new_doc)
                else:  # TEXT export
                    # Use text splitter
                    processed_docs = self.text_splitter.split_documents(docs)
                
                all_docs.extend(processed_docs)
                print(f"✓ Added {len(processed_docs)} chunks from {os.path.basename(pdf_path)}")
                
            except Exception as e:
                print(f"✗ Error processing {pdf_path}: {e}")
                continue
        
        # Filter complex metadata from all documents before storing
        if all_docs:
            print("Filtering complex metadata...")
            all_docs = filter_complex_metadata(all_docs)
            
            # Handle long documents that might cause embedding issues
            print("Checking for long documents...")
            all_docs = self._truncate_long_documents(all_docs)
        
        self.documents = all_docs
        print(f"\nTotal documents processed: {len(all_docs)}")
        return all_docs

    def build_vector_store(self, documents: Optional[List[Document]] = None):
        """
        Build vector store from documents
        
        Args:
            documents: List of documents (uses self.documents if None)
        """
        if documents is None:
            documents = self.documents
            
        if not documents:
            raise ValueError("No documents to index. Extract documents first.")
        
        print(f"Building vector store with {len(documents)} documents...")
        
        # Ensure complex metadata is filtered
        documents = filter_complex_metadata(documents)
        
        # Create vector store using Chroma
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="docling_rag"
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        print("✓ Vector store built successfully!")

    def add_pdf_to_index(self, pdf_path: str):
        """
        Add a single PDF to the existing index
        
        Args:
            pdf_path: Path to PDF file
        """
        new_docs = self.extract_documents_from_pdfs([pdf_path])
        
        if new_docs:
            # Filter complex metadata
            new_docs = filter_complex_metadata(new_docs)
            
            if self.vectorstore is None:
                # Create new vector store
                self.build_vector_store(new_docs)
            else:
                # Add to existing vector store
                self.vectorstore.add_documents(new_docs)
                print(f"✓ Added {len(new_docs)} new documents to existing index")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not built. Call build_vector_store() first.")
        
        # Update retriever with new top_k
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        # Perform similarity search with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                'text': doc.page_content,
                'source': doc.metadata.get('source_file', 'Unknown'),
                'score': float(1 - score),  # Convert distance to similarity score
                'metadata': doc.metadata
            })
        
        return results

    def setup_rag_chain(self, llm, prompt_template: Optional[str] = None):
        """
        Set up RAG chain with LLM
        
        Args:
            llm: LangChain LLM instance
            prompt_template: Custom prompt template
        """
        if self.retriever is None:
            raise ValueError("Retriever not set up. Call build_vector_store() first.")
        
        # Default prompt template
        if prompt_template is None:
            prompt_template = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {input}
Answer:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        
        return self.rag_chain

    def ask(self, question: str) -> Dict:
        """
        Ask a question using the RAG chain
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with answer and context
        """
        if not hasattr(self, 'rag_chain'):
            raise ValueError("RAG chain not set up. Call setup_rag_chain() first.")
        
        response = self.rag_chain.invoke({"input": question})
        return response

    def save_index(self, path: str):
        """Save the RAG system state"""
        # Vector store is automatically persisted if persist_directory is set
        # Save additional metadata
        metadata = {
            'embedding_model_name': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'export_type': self.export_type,
            'persist_directory': self.persist_directory,
            'num_documents': len(self.documents)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Index metadata saved to {path}")
        print(f"✓ Vector store persisted to {self.persist_directory}")

    def load_existing_vectorstore(self) -> bool:
        """
        Try to load existing vector store from persist directory
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="docling_rag"
                )
                self.retriever = self.vectorstore.as_retriever()
                
                # Get collection info
                collection = self.vectorstore._collection
                count = collection.count()
                print(f"✓ Loaded existing vector store with {count} documents")
                return True
            else:
                print("No existing vector store found")
                return False
        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            return False

    def get_processed_files(self) -> set:
        """Get list of files that have already been processed"""
        if self.vectorstore is None:
            return set()
        
        try:
            # Get all metadata to find processed files
            collection = self.vectorstore._collection
            results = collection.get(include=['metadatas'])
            
            processed_files = set()
            for metadata in results['metadatas']:
                if 'source_file' in metadata:
                    processed_files.add(metadata['source_file'])
            
            return processed_files
        except Exception as e:
            print(f"Error getting processed files: {e}")
            return set()

    def process_pdfs_incrementally(self, pdf_paths: List[str]) -> List[Document]:
        """
        Process only new PDFs that haven't been processed before
        """
        # Try to load existing vector store first
        self.load_existing_vectorstore()
        
        # Get already processed files
        processed_files = self.get_processed_files()
        
        # Filter out already processed files
        new_pdf_paths = []
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            if filename not in processed_files:
                new_pdf_paths.append(pdf_path)
            else:
                print(f"Skipping already processed: {filename}")
        
        if not new_pdf_paths:
            print("All PDFs already processed!")
            return []
        
        print(f"Processing {len(new_pdf_paths)} new PDFs (skipped {len(processed_files)} already processed)")
        
        # Extract documents from new PDFs only
        new_docs = self.extract_documents_from_pdfs(new_pdf_paths)
        
        # Add to existing or create new vector store
        if new_docs:
            if self.vectorstore is None:
                self.build_vector_store(new_docs)
            else:
                # Filter complex metadata and add to existing store
                new_docs = filter_complex_metadata(new_docs)
                new_docs = self._truncate_long_documents(new_docs)
                self.vectorstore.add_documents(new_docs)
                self.documents.extend(new_docs)
                print(f"✓ Added {len(new_docs)} documents to existing vector store")
        
        return new_docs
        """
        Inspect metadata of a document to understand its structure
        
        Args:
            doc_index: Index of document to inspect
        """
        if not self.documents or doc_index >= len(self.documents):
            print("No documents available or invalid index")
            return
        
        doc = self.documents[doc_index]
        print(f"Document {doc_index} metadata:")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Metadata keys: {list(doc.metadata.keys())}")
        
        for key, value in doc.metadata.items():
            if isinstance(value, (dict, list)):
                print(f"  {key}: {type(value).__name__} (complex)")
                if isinstance(value, dict):
                    print(f"    Keys: {list(value.keys())}")
                elif isinstance(value, list) and value:
                    print(f"    Length: {len(value)}, First item type: {type(value[0]).__name__}")
            else:
                print(f"  {key}: {value} ({type(value).__name__})")

    def inspect_metadata(self, doc_index: int = 0):
        """
        Inspect metadata of a document to understand its structure
        
        Args:
            doc_index: Index of document to inspect
        """
        if not self.documents or doc_index >= len(self.documents):
            print("No documents available or invalid index")
            return
        
        doc = self.documents[doc_index]
        print(f"Document {doc_index} metadata:")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Metadata keys: {list(doc.metadata.keys())}")
        
        for key, value in doc.metadata.items():
            if isinstance(value, (dict, list)):
                print(f"  {key}: {type(value).__name__} (complex)")
                if isinstance(value, dict):
                    print(f"    Keys: {list(value.keys())}")
                elif isinstance(value, list) and value:
                    print(f"    Length: {len(value)}, First item type: {type(value[0]).__name__}")
            else:
                print(f"  {key}: {value} ({type(value).__name__})")

    def get_stats(self):
        """Get statistics about the RAG system"""
        stats = {
            'total_documents': len(self.documents),
            'vectorstore_size': 0,
            'processed_files': set()
        }
        
        if self.vectorstore:
            try:
                collection = self.vectorstore._collection
                stats['vectorstore_size'] = collection.count()
                
                # Get processed files
                results = collection.get(include=['metadatas'])
                for metadata in results['metadatas']:
                    if 'source_file' in metadata:
                        stats['processed_files'].add(metadata['source_file'])
            except:
                pass
        
        return stats

    def load_index(self, path: str):
        """Load the RAG system state"""
        with open(path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Restore configuration
        self.embedding_model_name = metadata['embedding_model_name']
        self.chunk_size = metadata['chunk_size']
        self.chunk_overlap = metadata['chunk_overlap']
        self.export_type = metadata['export_type']
        self.persist_directory = metadata['persist_directory']
        
        # Reinitialize embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Load existing vector store
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="docling_rag"
            )
            self.retriever = self.vectorstore.as_retriever()
            print(f"✓ Index loaded from {path}")
            print(f"✓ Vector store loaded from {self.persist_directory}")
        else:
            print(f"Warning: Vector store directory {self.persist_directory} not found")


def main():
    """Example usage of the LangChain Docling RAG system with incremental processing"""
    
    # Initialize RAG system with optimized settings
    rag = LangChainDoclingRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=800,  # Larger chunks to reduce excessive splitting
        chunk_overlap=100,
        export_type=ExportType.DOC_CHUNKS,
        persist_directory="./chroma_db",
        max_token_length=450  # Safe limit for embedding model
    )
    
    # Process PDFs
    pdf_folder = "/home/karang/dikshant/extramarks_239/rag/books"
    
    if os.path.exists(pdf_folder):
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        
        # Process all PDFs (or limit for testing)
        max_pdfs = 15  # Set to None to process all
        pdf_paths = [str(pdf) for pdf in pdf_files[:max_pdfs]] if max_pdfs else [str(pdf) for pdf in pdf_files]
        
        print(f"Found {len(pdf_files)} PDF files, processing {len(pdf_paths)}...")
        
        # Use incremental processing (will skip already processed files)
        new_documents = rag.process_pdfs_incrementally(pdf_paths)
        
        # Show statistics
        stats = rag.get_stats()
        print(f"\n{'='*50}")
        print("RAG System Statistics:")
        print(f"{'='*50}")
        print(f"Total documents in memory: {stats['total_documents']}")
        print(f"Vector store size: {stats['vectorstore_size']}")
        print(f"Processed files: {len(stats['processed_files'])}")
        print(f"Files: {', '.join(sorted(stats['processed_files']))}")
        
        # Save metadata (vector store auto-persists)
        rag.save_index("rag_metadata.pkl")
        
        # Test search only if we have documents
        if rag.vectorstore:
            test_query = "What did Liberal Nationalism Stand for?"
            print(f"\n{'='*50}")
            print(f"Search Query: '{test_query}'")
            print(f"{'='*50}")
            
            results = rag.search(test_query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i} (Score: {result['score']:.3f})")
                print(f"Source: {result['source']}")
                # Show more context
                text = result['text']
                if len(text) > 300:
                    print(f"Text: {text[:300]}...")
                else:
                    print(f"Text: {text}")
        else:
            print("No vector store available for search")



def quick_search_example():
    """Quick example showing how to load and search existing index"""
    print("Loading existing RAG system...")
    
    rag = LangChainDoclingRAG(persist_directory="./chroma_db")
    
    if rag.load_existing_vectorstore():
        # Search without reprocessing
        query = "What did Liberal Nationalism Stand for?"
        results = rag.search(query, top_k=3)
        
        print(f"Search results for: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}: {result['text'][:200]}...")
    else:
        print("No existing vector store found. Run main() first.")


if __name__ == "__main__":
    # For first time setup with all PDFs
    main()
    
    # For subsequent searches (uncomment to test)
    # quick_search_example()