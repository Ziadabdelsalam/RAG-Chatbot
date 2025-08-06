import os
from typing import List, Dict, Optional
from rag_system import RAGSystem, RAGSystemError
import time
import logging

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks import get_openai_callback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotError(Exception):
    """Custom exception for chatbot errors"""
    pass

class RAGChatbot:
    def __init__(self, rag_system: RAGSystem, model_name: str = "gpt-4o-mini", 
                 max_tokens: int = 1000, temperature: float = 0.7):
        """
        Initialize the RAG Chatbot with LangChain ChatOpenAI
        
        Args:
            rag_system: Instance of RAGSystem for document retrieval
            model_name: OpenAI model to use for generation
            max_tokens: Maximum tokens for response generation
            temperature: Temperature for response generation
        """
        self.rag_system = rag_system
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize LangChain ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ChatbotError("OPENAI_API_KEY environment variable is not set")
        
        try:
            self.llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout=60,
                max_retries=3
            )
            logger.info(f"LangChain ChatOpenAI initialized with model: {model_name}")
            
        except Exception as e:
            raise ChatbotError(f"Failed to initialize LangChain ChatOpenAI: {str(e)}")
        
        # Conversation management
        self.conversation_history = []
        self.max_history_length = 20  # Keep only last 20 messages
        self.max_context_length = 6   # Use last 6 messages for context
        
        # Performance tracking
        self.response_times = []
        self.api_call_count = 0
        self.error_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
    
    def generate_response(self, user_query: str, use_rag: bool = True, 
                         n_context_docs: int = 3, min_similarity: float = 0.3) -> Dict:
        """
        Generate a response using LangChain ChatOpenAI with enhanced tracking
        
        Args:
            user_query: The user's question
            use_rag: Whether to use RAG for context retrieval
            n_context_docs: Number of context documents to retrieve
            min_similarity: Minimum similarity threshold for retrieved documents
            
        Returns:
            Dictionary containing response and reference documents
        """
        start_time = time.time()
        
        try:
            # Input validation
            if not user_query or not user_query.strip():
                return {
                    "response": "I'm sorry, but I didn't receive a valid question. Could you please rephrase your query?",
                    "references": [],
                    "used_rag": False,
                    "processing_time": 0.0,
                    "tokens_used": 0,
                    "cost": 0.0
                }
            
            # Preprocess user query
            processed_query = self._preprocess_user_query(user_query)
            
            # Retrieve relevant context if RAG is enabled
            context = ""
            search_results = None
            reference_docs = []
            
            if use_rag:
                try:
                    search_results = self.rag_system.search_documents(
                        processed_query, 
                        n_results=n_context_docs,
                        min_similarity=min_similarity
                    )
                    
                    if search_results["documents"]:
                        context_docs = search_results["documents"]
                        context = self._format_context(context_docs)
                        reference_docs = self._prepare_reference_docs(search_results)
                        logger.info(f"Retrieved {len(context_docs)} relevant documents")
                    else:
                        logger.info("No relevant documents found for the query")
                        
                except Exception as e:
                    logger.error(f"RAG retrieval failed: {str(e)}")
                    # Continue without RAG if retrieval fails
                    use_rag = False
            
            # Generate response using LangChain
            response_text, tokens_used, cost = self._generate_llm_response(processed_query, context, use_rag)
            
            # Update conversation history
            self._update_conversation_history(user_query, response_text)
            
            # Track performance
            processing_time = time.time() - start_time
            self.response_times.append(processing_time)
            self.api_call_count += 1
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            return {
                "response": response_text,
                "references": reference_docs,
                "used_rag": use_rag,
                "processing_time": round(processing_time, 2),
                "context_docs_found": len(reference_docs),
                "tokens_used": tokens_used,
                "cost": round(cost, 6)
            }
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question.",
                "references": [],
                "used_rag": False,
                "processing_time": time.time() - start_time,
                "tokens_used": 0,
                "cost": 0.0,
                "error": str(e)
            }
    
    def _preprocess_user_query(self, query: str) -> str:
        """Preprocess and clean user query"""
        # Basic cleaning
        query = query.strip()
        
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        return query
    
    def _format_context(self, context_docs: List[str]) -> str:
        """Format retrieved documents into context string"""
        if not context_docs:
            return ""
        
        formatted_context = []
        for i, doc in enumerate(context_docs, 1):
            # Truncate very long documents to fit in context window
            truncated_doc = doc[:1500] + "..." if len(doc) > 1500 else doc
            formatted_context.append(f"Context {i}:\n{truncated_doc}")
        
        return "\n\n".join(formatted_context)
    
    def _prepare_reference_docs(self, search_results: Dict) -> List[Dict]:
        """Prepare reference documents for display"""
        reference_docs = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            search_results["documents"],
            search_results["metadatas"],
            search_results["distances"]
        )):
            source = metadata.get("source", "Unknown")
            chunk_id = metadata.get("chunk_id", "")
            similarity = round((1 - distance) * 100, 1)
            
            # Create preview content
            content_preview = doc[:200] + "..." if len(doc) > 200 else doc
            
            reference_docs.append({
                "source": source,
                "chunk_id": chunk_id,
                "content": content_preview,
                "similarity": similarity,
                "full_content": doc,
                "metadata": metadata
            })
        
        return reference_docs
    
    def _generate_llm_response(self, query: str, context: str, used_rag: bool) -> tuple:
        """Generate response using LangChain ChatOpenAI with token tracking"""
        try:
            # Prepare messages
            messages = []
            
            # System message
            system_prompt = self._get_system_prompt(context, used_rag)
            messages.append(SystemMessage(content=system_prompt))
            
            # Add relevant conversation history
            relevant_history = self._get_relevant_history()
            for msg in relevant_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current user query
            messages.append(HumanMessage(content=query))
            
            # Generate response with token tracking
            with get_openai_callback() as cb:
                response = self.llm(messages)
                
                assistant_response = response.content
                tokens_used = cb.total_tokens
                total_cost = cb.total_cost
            
            if not assistant_response:
                raise ChatbotError("Received empty response from LangChain ChatOpenAI")
            
            return assistant_response.strip(), tokens_used, total_cost
            
        except Exception as e:
            logger.error(f"LangChain API call failed: {str(e)}")
            raise ChatbotError(f"Failed to generate response: {str(e)}")
    
    def _get_system_prompt(self, context: str = "", used_rag: bool = False) -> str:
        """
        Get the system prompt for the chatbot
        
        Args:
            context: Retrieved context from the knowledge base
            used_rag: Whether RAG was used for this query
            
        Returns:
            System prompt string
        """
        base_prompt = """You are a helpful AI assistant that provides accurate and informative responses. 

Guidelines for your responses:
1. Be clear, concise, and helpful
2. If you're uncertain about something, acknowledge the uncertainty
3. Provide structured responses when appropriate
4. Stay focused on the user's question
5. Be conversational but professional"""
        
        if used_rag and context:
            return f"""{base_prompt}

IMPORTANT: You have access to relevant context from a knowledge base. Use this context as your primary source of information for answering the user's question.

CONTEXT FROM KNOWLEDGE BASE:
{context}

Instructions:
- Prioritize information from the provided context
- If the context fully answers the question, base your response primarily on it
- If the context is incomplete, you may supplement with general knowledge but clearly indicate what comes from the context vs. your general knowledge
- If the context is not relevant to the question, let the user know and provide a general response
- Always be transparent about your sources"""
        else:
            return f"""{base_prompt}

Note: You are responding based on your general knowledge as no specific context was retrieved from the knowledge base."""
    
    def _get_relevant_history(self) -> List[Dict]:
        """Get relevant conversation history for context"""
        # Return the last few exchanges, but limit total length
        relevant_history = self.conversation_history[-self.max_context_length:]
        
        # Ensure we don't exceed token limits by truncating if necessary
        total_length = sum(len(msg["content"]) for msg in relevant_history)
        if total_length > 2000:  # Approximate token limit
            # Keep only the most recent exchanges
            relevant_history = self.conversation_history[-4:]
        
        return relevant_history
    
    def _update_conversation_history(self, user_query: str, assistant_response: str):
        """Update conversation history with new exchange"""
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Trim conversation history if it gets too long
        if len(self.conversation_history) > self.max_history_length:
            # Remove oldest exchanges but keep pairs
            excess = len(self.conversation_history) - self.max_history_length
            # Ensure we remove complete exchanges (pairs)
            if excess % 2 == 1:
                excess += 1
            self.conversation_history = self.conversation_history[excess:]
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the current conversation history"""
        return self.conversation_history.copy()
    
    def add_documents_from_text(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents to the knowledge base from text with enhanced error handling
        
        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document
        """
        if not texts:
            raise ChatbotError("No texts provided for adding to knowledge base")
        
        try:
            self.rag_system.add_text_documents(texts, metadatas)
            logger.info(f"Successfully added {len(texts)} text documents to knowledge base")
        except RAGSystemError as e:
            raise ChatbotError(f"Failed to add text documents: {str(e)}")
        except Exception as e:
            raise ChatbotError(f"Unexpected error adding text documents: {str(e)}")
    
    def add_documents_from_pdf(self, pdf_paths: List[str]):
        """
        Add documents to the knowledge base from PDF files with enhanced error handling
        
        Args:
            pdf_paths: List of paths to PDF files or file-like objects
        """
        if not pdf_paths:
            raise ChatbotError("No PDF files provided")
        
        successful_adds = []
        failed_adds = []
        
        for pdf_path in pdf_paths:
            try:
                self.rag_system.add_pdf_document(pdf_path)
                successful_adds.append(pdf_path)
                logger.info(f"Successfully added PDF: {pdf_path}")
            except RAGSystemError as e:
                failed_adds.append((pdf_path, str(e)))
                logger.error(f"Failed to add PDF '{pdf_path}': {str(e)}")
            except Exception as e:
                failed_adds.append((pdf_path, str(e)))
                logger.error(f"Unexpected error adding PDF '{pdf_path}': {str(e)}")
        
        if failed_adds and not successful_adds:
            raise ChatbotError(f"Failed to add all PDF documents: {failed_adds}")
        elif failed_adds:
            logger.warning(f"Partially successful: {len(successful_adds)} succeeded, {len(failed_adds)} failed")
    
    def search_knowledge_base(self, query: str, n_results: int = 5, 
                             min_similarity: float = 0.0) -> Dict:
        """
        Search the knowledge base directly with enhanced parameters
        
        Args:
            query: Search query
            n_results: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            Search results with additional metadata
        """
        try:
            results = self.rag_system.search_documents(query, n_results, min_similarity)
            
            # Enhance results with additional information
            enhanced_results = {
                "query": query,
                "total_results": len(results.get("documents", [])),
                "results": []
            }
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results.get("documents", []),
                results.get("metadatas", []),
                results.get("distances", [])
            )):
                enhanced_results["results"].append({
                    "rank": i + 1,
                    "document": doc,
                    "metadata": metadata,
                    "similarity": round((1 - distance) * 100, 1),
                    "source": metadata.get("source", "Unknown")
                })
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {str(e)}")
            return {"query": query, "total_results": 0, "results": [], "error": str(e)}
    
    def get_knowledge_base_info(self) -> Dict:
        """Get comprehensive information about the knowledge base"""
        try:
            base_info = self.rag_system.get_collection_info()
            
            # Add chatbot-specific statistics
            performance_stats = {
                "total_api_calls": self.api_call_count,
                "total_errors": self.error_count,
                "average_response_time": round(sum(self.response_times) / len(self.response_times), 2) if self.response_times else 0,
                "conversation_length": len(self.conversation_history) // 2,  # Number of exchanges
                "total_tokens_used": self.total_tokens_used,
                "total_cost": round(self.total_cost, 4),
                "model_name": self.model_name
            }
            
            return {**base_info, "performance_stats": performance_stats}
            
        except Exception as e:
            logger.error(f"Error getting knowledge base info: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics"""
        if not self.response_times:
            return {"message": "No performance data available yet"}
        
        return {
            "total_responses": len(self.response_times),
            "total_api_calls": self.api_call_count,
            "total_errors": self.error_count,
            "average_response_time": round(sum(self.response_times) / len(self.response_times), 2),
            "min_response_time": round(min(self.response_times), 2),
            "max_response_time": round(max(self.response_times), 2),
            "success_rate": round((self.api_call_count - self.error_count) / self.api_call_count * 100, 1) if self.api_call_count > 0 else 0,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": round(self.total_cost, 4),
            "average_cost_per_call": round(self.total_cost / self.api_call_count, 6) if self.api_call_count > 0 else 0,
            "model_name": self.model_name
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics"""
        self.response_times = []
        self.api_call_count = 0
        self.error_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        logger.info("Performance statistics reset")