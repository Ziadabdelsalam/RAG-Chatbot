import os
# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import PyPDF2
from pathlib import Path
import hashlib
import logging
from contextlib import contextmanager
import tempfile
import time
import re

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystemError(Exception):
    """Custom exception for RAG system errors"""
    pass

class LangChainEmbeddingFunction:
    """Wrapper to make LangChain embeddings work with ChromaDB"""
    
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings
    
    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        """Convert LangChain embeddings to ChromaDB format"""
        try:
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            
            embeddings = self.langchain_embeddings.embed_documents(input_texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

class RAGSystem:
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system with ChromaDB and LangChain OpenAI embeddings
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
        self.supported_extensions = {'.pdf', '.txt', '.md'}
        
        # Initialize ChromaDB client
        try:
            import chromadb.config
            settings = chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=settings
            )
            logger.info(f"ChromaDB client initialized with path: {persist_directory}")
        except Exception as e:
            raise RAGSystemError(f"Failed to initialize ChromaDB client: {str(e)}")
        
        # Initialize LangChain OpenAI embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RAGSystemError("OPENAI_API_KEY environment variable is not set")
        
        try:
            # Use LangChain OpenAI embeddings
            self.langchain_embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-3-small",
                openai_api_base=None,  # Use default
                chunk_size=1000,  # Process embeddings in chunks
                max_retries=3
            )
            
            # Create wrapper for ChromaDB
            self.embedding_function = LangChainEmbeddingFunction(self.langchain_embeddings)
            
            logger.info("LangChain OpenAI embeddings initialized successfully")
        except Exception as e:
            raise RAGSystemError(f"Failed to initialize LangChain embeddings: {str(e)}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Get or create collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the ChromaDB collection"""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Created new collection: {self.collection_name}")
            except Exception as e:
                raise RAGSystemError(f"Failed to create collection: {str(e)}")
    
    def validate_file(self, file_path: str) -> bool:
        """Validate file before processing"""
        if not os.path.exists(file_path):
            raise RAGSystemError(f"File does not exist: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise RAGSystemError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_extensions:
            raise RAGSystemError(f"Unsupported file type: {file_ext}")
        
        return True
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for content to detect duplicates"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _check_duplicate_content(self, content_hash: str) -> bool:
        """Check if content already exists in the collection"""
        try:
            results = self.collection.get(where={"content_hash": content_hash})
            return len(results['ids']) > 0
        except Exception:
            return False
    
    def add_text_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None, 
                          ids: Optional[List[str]] = None):
        """
        Add text documents to the vector database using LangChain text splitting
        
        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
        """
        if not texts:
            raise RAGSystemError("No texts provided")
        
        if ids is None:
            ids = [f"doc_{i}_{int(time.time())}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(texts))]
        
        try:
            all_chunked_texts = []
            all_chunked_metadatas = []
            all_chunked_ids = []
            
            for i, text in enumerate(texts):
                # Check for duplicates
                content_hash = self._generate_content_hash(text)
                if self._check_duplicate_content(content_hash):
                    logger.warning(f"Duplicate content detected for document {ids[i]}, skipping")
                    continue
                
                # Create LangChain Document
                doc = Document(page_content=text, metadata=metadatas[i])
                
                # Split using LangChain text splitter
                chunks = self.text_splitter.split_documents([doc])
                
                for j, chunk in enumerate(chunks):
                    all_chunked_texts.append(chunk.page_content)
                    
                    # Combine original metadata with chunk info
                    chunk_metadata = {
                        **chunk.metadata,
                        "chunk_id": j,
                        "content_hash": content_hash,
                        "total_chunks": len(chunks)
                    }
                    all_chunked_metadatas.append(chunk_metadata)
                    all_chunked_ids.append(f"{ids[i]}_chunk_{j}")
            
            if all_chunked_texts:
                # Add to ChromaDB collection
                self.collection.add(
                    documents=all_chunked_texts,
                    metadatas=all_chunked_metadatas,
                    ids=all_chunked_ids
                )
                logger.info(f"Added {len(all_chunked_texts)} document chunks using LangChain")
            else:
                logger.info("No new documents to add (all duplicates)")
                
        except Exception as e:
            raise RAGSystemError(f"Failed to add documents: {str(e)}")
    
    @contextmanager
    def _safe_file_processing(self, file_path):
        """Context manager for safe file processing"""
        temp_file = None
        try:
            if hasattr(file_path, 'read'):  # File-like object
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                file_path.seek(0)
                temp_file.write(file_path.read())
                temp_file.close()
                yield temp_file.name
            else:
                yield file_path
        finally:
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def add_pdf_document(self, pdf_path, metadata: Optional[Dict] = None):
        """
        Add a PDF document to the vector database with LangChain processing
        
        Args:
            pdf_path: Path to the PDF file or file-like object
            metadata: Optional metadata for the document
        """
        try:
            with self._safe_file_processing(pdf_path) as safe_path:
                if isinstance(pdf_path, str):
                    self.validate_file(safe_path)
                    file_name = Path(safe_path).stem
                else:
                    file_name = getattr(pdf_path, 'name', 'uploaded_pdf')
                    if file_name.startswith('/'):
                        file_name = Path(file_name).stem
                
                text = self._extract_text_from_pdf(safe_path)
                if not text.strip():
                    raise RAGSystemError("No text could be extracted from PDF")
                
                if metadata is None:
                    metadata = {
                        "source": file_name,
                        "type": "pdf",
                        "upload_time": time.time()
                    }
                
                doc_id = f"pdf_{file_name}_{int(time.time())}"
                self.add_text_documents([text], [metadata], [doc_id])
                logger.info(f"Successfully added PDF: {file_name}")
                
        except Exception as e:
            raise RAGSystemError(f"Failed to process PDF: {str(e)}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file with better error handling"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if len(pdf_reader.pages) == 0:
                    raise RAGSystemError("PDF has no pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                        continue
                        
        except Exception as e:
            raise RAGSystemError(f"Error reading PDF file: {str(e)}")
        
        return text.strip()
    
    def search_documents(self, query: str, n_results: int = 5, 
                        min_similarity: float = 0.0) -> Dict:
        """
        Search for relevant documents using LangChain embeddings
        
        Args:
            query: The search query
            n_results: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            Dictionary containing the search results
        """
        if not query.strip():
            return {"documents": [], "metadatas": [], "distances": []}
        
        try:
            # Use ChromaDB's query method with LangChain embeddings
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Filter by similarity if specified
            if min_similarity > 0.0:
                filtered_results = self._filter_by_similarity(results, min_similarity, n_results)
                return filtered_results
            
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    def _filter_by_similarity(self, results: Dict, min_similarity: float, n_results: int) -> Dict:
        """Filter results by similarity threshold"""
        if not results["documents"] or not results["documents"][0]:
            return {"documents": [], "metadatas": [], "distances": []}
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        # Filter by similarity (convert distance to similarity)
        filtered_docs = []
        filtered_metadata = []
        filtered_distances = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1 - dist  # Convert distance to similarity
            if similarity >= min_similarity:
                filtered_docs.append(doc)
                filtered_metadata.append(meta)
                filtered_distances.append(dist)
        
        # Return top n_results
        return {
            "documents": filtered_docs[:n_results],
            "metadatas": filtered_metadata[:n_results],
            "distances": filtered_distances[:n_results]
        }
    
    def get_collection_info(self) -> Dict:
        """Get comprehensive information about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "max_file_size_mb": self.max_file_size / (1024 * 1024),
                "supported_extensions": list(self.supported_extensions),
                "embedding_model": "text-embedding-3-small (LangChain)",
                "text_splitter": "RecursiveCharacterTextSplitter"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }
    
    def clear_collection(self):
        """Clear all documents from the collection efficiently"""
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs['ids']:
                # Delete in batches
                batch_size = 100
                for i in range(0, len(all_docs['ids']), batch_size):
                    batch_ids = all_docs['ids'][i:i + batch_size]
                    self.collection.delete(ids=batch_ids)
                    logger.info(f"Deleted batch {i//batch_size + 1}")
            
            logger.info("Collection cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            # Fallback to recreation if deletion fails
            try:
                self.client.delete_collection(name=self.collection_name)
                self._initialize_collection()
                logger.info("Collection recreated successfully")
            except Exception as e2:
                raise RAGSystemError(f"Failed to clear collection: {str(e2)}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a specific document by ID"""
        try:
            result = self.collection.get(ids=[doc_id])
            if result['documents']:
                return {
                    "id": doc_id,
                    "document": result['documents'][0],
                    "metadata": result['metadatas'][0] if result['metadatas'] else {}
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None