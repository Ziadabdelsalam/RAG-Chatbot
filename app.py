import os
import sys
import traceback
from pathlib import Path

# Set environment variable before any imports
os.environ["PYDANTIC_V2"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

try:
    import gradio as gr
    from dotenv import load_dotenv
    import tempfile
    import logging
    from typing import List, Tuple, Optional
    import time
    
    # Load environment variables
    load_dotenv()
    
    from rag_system import RAGSystem, RAGSystemError
    from chatbot import RAGChatbot, ChatbotError
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
rag_system = None
chatbot = None

def initialize_app():
    """Initialize the RAG system and chatbot."""
    global rag_system, chatbot
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY not found in environment variables"
        
        logger.info("Initializing RAG system with LangChain...")
        rag_system = RAGSystem()
        
        logger.info("Initializing chatbot with LangChain ChatOpenAI...")
        chatbot = RAGChatbot(rag_system, model_name="gpt-4o-mini")
        
        return True, "Application initialized successfully with LangChain!"
        
    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg

def format_response_with_references(result: dict) -> str:
    """Formats the chatbot's response to include reference documents with enhanced styling."""
    if not isinstance(result, dict):
        return str(result)
    
    response_text = result.get("response", "No response generated.")
    references = result.get("references", [])
    processing_time = result.get("processing_time", 0)
    used_rag = result.get("used_rag", False)
    context_docs_found = result.get("context_docs_found", 0)
    tokens_used = result.get("tokens_used", 0)
    cost = result.get("cost", 0.0)
    
    # Add metadata info
    metadata_info = f"\n\n---\n**📊 Response Metadata (LangChain):**\n"
    metadata_info += f"- Processing Time: {processing_time}s\n"
    metadata_info += f"- Used RAG: {'Yes' if used_rag else 'No'}\n"
    metadata_info += f"- Context Documents Found: {context_docs_found}\n"
    metadata_info += f"- Tokens Used: {tokens_used}\n"
    metadata_info += f"- Cost: ${cost:.6f}\n"
    
    if references:
        reference_text = "\n\n---\n**📚 Reference Documents:**\n"
        for i, ref in enumerate(references, 1):
            source = ref.get('source', 'Unknown')
            similarity = ref.get('similarity', 'N/A')
            chunk_id = ref.get('chunk_id', '')
            content = ref.get('content', 'No content available.')
            
            chunk_info = f" (Chunk {chunk_id})" if chunk_id != '' else ""
            reference_text += f"\n**{i}. {source}{chunk_info}** - Similarity: {similarity}%\n"
            reference_text += f"```\n{content}\n```\n"
        
        response_text += reference_text
    
    response_text += metadata_info
    return response_text

def handle_chat(message: str, history: List[Tuple[str, str]], use_rag: bool, 
               min_similarity: float) -> Tuple[str, List[Tuple[str, str]]]:
    """Handle chat messages with LangChain."""
    if not message.strip():
        return "", history
    
    if chatbot is None:
        history.append((message, "❌ Chatbot not initialized"))
        return "", history

    try:
        result = chatbot.generate_response(
            message, 
            use_rag=use_rag, 
            min_similarity=min_similarity
        )
        
        response_text = format_response_with_references(result)
        history.append((message, response_text))
        return "", history
        
    except Exception as e:
        error_response = f"❌ Error: {str(e)}"
        history.append((message, error_response))
        return "", history

def handle_upload(files):
    """Handle file uploads."""
    if not files:
        return "No files selected"
    
    if chatbot is None:
        return "❌ Chatbot not initialized"
    
    try:
        success_count = 0
        error_count = 0
        
        for file_obj in files:
            try:
                file_name = file_obj.name
                file_ext = Path(file_name).suffix.lower()
                
                if file_ext == '.pdf':
                    chatbot.add_documents_from_pdf([file_obj])
                    success_count += 1
                elif file_ext in ['.txt', '.md']:
                    content = file_obj.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='ignore')
                    
                    chatbot.add_documents_from_text(
                        [content], 
                        [{"source": file_name, "type": "text"}]
                    )
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {file_obj.name}: {e}")
                error_count += 1
        
        return f"✅ Processed {success_count} files with LangChain. ❌ {error_count} errors."
        
    except Exception as e:
        return f"❌ Upload error: {str(e)}"

def get_stats():
    """Get knowledge base statistics."""
    if rag_system is None:
        return "❌ System not initialized"
    
    try:
        stats = rag_system.get_collection_info()
        kb_info = chatbot.get_knowledge_base_info() if chatbot else stats
        
        info_text = f"""📊 **Knowledge Base Stats (LangChain):**
- **Collection Name:** {kb_info.get('collection_name', 'N/A')}
- **Document Count:** {kb_info.get('document_count', 0):,}
- **Storage Directory:** {kb_info.get('persist_directory', 'N/A')}
- **Embedding Model:** {kb_info.get('embedding_model', 'N/A')}
- **Text Splitter:** {kb_info.get('text_splitter', 'N/A')}"""

        # Add performance stats if available
        if 'performance_stats' in kb_info:
            perf = kb_info['performance_stats']
            info_text += f"""

**🚀 Performance Statistics:**
- **Model:** {perf.get('model_name', 'N/A')}
- **Total API Calls:** {perf.get('total_api_calls', 0):,}
- **Total Tokens Used:** {perf.get('total_tokens_used', 0):,}
- **Total Cost:** ${perf.get('total_cost', 0):.4f}
- **Average Response Time:** {perf.get('average_response_time', 0):.2f}s
- **Conversation Exchanges:** {perf.get('conversation_length', 0):,}"""

        return info_text
        
    except Exception as e:
        logger.error(f"Error getting KB stats: {str(e)}")
        return f"❌ Error retrieving statistics: {str(e)}"

def get_performance_stats() -> str:
    """Get detailed performance statistics."""
    if chatbot is None:
        return "❌ Chatbot is not initialized."
    
    try:
        stats = chatbot.get_performance_stats()
        if 'message' in stats:
            return stats['message']
        
        perf_text = f"""**📈 Performance Dashboard (LangChain):**
- **Model:** {stats.get('model_name', 'N/A')}
- **Total Responses Generated:** {stats['total_responses']:,}
- **Total API Calls:** {stats['total_api_calls']:,}
- **Total Tokens Used:** {stats['total_tokens_used']:,}
- **Total Cost:** ${stats['total_cost']:.4f}
- **Average Cost per Call:** ${stats['average_cost_per_call']:.6f}
- **Total Errors:** {stats['total_errors']:,}
- **Success Rate:** {stats['success_rate']}%
- **Average Response Time:** {stats['average_response_time']}s
- **Fastest Response:** {stats['min_response_time']}s
- **Slowest Response:** {stats['max_response_time']}s"""
        
        return perf_text
        
    except Exception as e:
        return f"❌ Error retrieving performance stats: {str(e)}"

def clear_kb():
    """Clear knowledge base."""
    if rag_system is None:
        return "❌ System not initialized"
    
    try:
        rag_system.clear_collection()
        return "✅ Knowledge base cleared"
    except Exception as e:
        return f"❌ Error clearing KB: {str(e)}"

def search_kb(query: str, n_results: int, min_similarity: float) -> str:
    """Search the knowledge base directly."""
    if not query.strip():
        return "Please enter a search query."
    
    if chatbot is None:
        return "❌ Chatbot is not initialized."
    
    try:
        results = chatbot.search_knowledge_base(query, n_results, min_similarity)
        
        if results.get('total_results', 0) == 0:
            return f"No results found for query: '{query}'"
        
        search_output = f"**🔍 Search Results (LangChain) for:** '{query}'\n"
        search_output += f"**Found {results['total_results']} results:**\n\n"
        
        for result in results['results']:
            search_output += f"**{result['rank']}. {result['source']}** (Similarity: {result['similarity']}%)\n"
            search_output += f"```\n{result['document'][:300]}{'...' if len(result['document']) > 300 else ''}\n```\n\n"
        
        return search_output
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"❌ Search error: {str(e)}"

# Initialize the application
init_success, init_message = initialize_app()

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="LangChain RAG Chatbot") as demo:
        
        if not init_success:
            gr.Markdown("# ❌ Initialization Error")
            gr.Markdown(f"**Error:** {init_message}")
            gr.Markdown("**Solutions:**")
            gr.Markdown("1. Check your .env file contains OPENAI_API_KEY")
            gr.Markdown("2. Run: `pip install -r requirements.txt`")
            gr.Markdown("3. Try creating a virtual environment")
            return demo
        
        gr.Markdown("# 🤖 LangChain RAG Chatbot")
        gr.Markdown("*Powered by LangChain, OpenAI GPT-4o-mini, ChromaDB and Gradio*")
        gr.Markdown("*By Ziad Ahmed*")
        
        with gr.Tab("💬 Chat"):
            chatbot_ui = gr.Chatbot(label="Conversation", height=500)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask me anything about your documents...", 
                    label="Message",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                use_rag = gr.Checkbox(label="Use RAG", value=True)
                min_similarity_slider = gr.Slider(
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.3, 
                    step=0.1,
                    label="Minimum Similarity"
                )
                clear_btn = gr.Button("Clear Chat")
        
        with gr.Tab("📁 Upload"):
            with gr.Column():
                gr.Markdown("### Upload Documents (LangChain Processing)")
                gr.Markdown("Supported: PDF, TXT, MD files | Uses RecursiveCharacterTextSplitter")
                
                file_input = gr.File(
                    label="Choose files",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md"]
                )
                upload_btn = gr.Button("Upload", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("🔍 Search"):
            with gr.Column():
                gr.Markdown("### Search Knowledge Base")
                search_input = gr.Textbox(placeholder="Search documents...", label="Search Query")
                with gr.Row():
                    search_results_slider = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1, label="Max Results"
                    )
                    search_similarity_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.3, step=0.1, label="Min Similarity"
                    )
                search_button = gr.Button("Search", variant="secondary")
                search_output = gr.Markdown("Enter a query to search your documents.")
        
        with gr.Tab("📊 Stats"):
            with gr.Column():
                stats_display = gr.Markdown(get_stats())
                refresh_btn = gr.Button("Refresh Stats")
                
                gr.Markdown("### Performance Dashboard")
                performance_display = gr.Markdown(get_performance_stats())
                refresh_performance_btn = gr.Button("🔄 Refresh Performance Stats")
                reset_stats_btn = gr.Button("🗑️ Reset Stats", variant="secondary")
        
        with gr.Tab("⚙️ Settings"):
            with gr.Column():
                gr.Markdown("### System Information (LangChain)")
                system_info = gr.Markdown(f"""
                - **OpenAI API Key:** {'✅ Configured' if os.getenv('OPENAI_API_KEY') else '❌ Not Set'}
                - **LLM Model:** gpt-4o-mini (LangChain)
                - **Embedding Model:** text-embedding-3-small (LangChain)
                - **Text Splitter:** RecursiveCharacterTextSplitter
                - **Max File Size:** 50MB
                """)
                
                gr.Markdown("### ⚠️ Danger Zone")
                clear_kb_btn = gr.Button("🗑️ Clear Knowledge Base", variant="stop")
                clear_status = gr.Textbox(label="Status", interactive=False)
        
        # Event handlers
        send_btn.click(
            handle_chat,
            inputs=[msg_input, chatbot_ui, use_rag, min_similarity_slider],
            outputs=[msg_input, chatbot_ui]
        )
        
        msg_input.submit(
            handle_chat,
            inputs=[msg_input, chatbot_ui, use_rag, min_similarity_slider],
            outputs=[msg_input, chatbot_ui]
        )
        
        clear_btn.click(lambda: [], outputs=chatbot_ui)
        
        upload_btn.click(
            handle_upload,
            inputs=file_input,
            outputs=upload_status
        )
        
        search_button.click(
            search_kb,
            inputs=[search_input, search_results_slider, search_similarity_slider],
            outputs=search_output
        )
        
        refresh_btn.click(get_stats, outputs=stats_display)
        refresh_performance_btn.click(get_performance_stats, outputs=performance_display)
        reset_stats_btn.click(
            lambda: (chatbot.reset_performance_stats() if chatbot else None, "📊 Performance statistics reset.")[1], 
            outputs=performance_display
        )
        
        clear_kb_btn.click(clear_kb, outputs=clear_status)
    
    return demo

if __name__ == "__main__":
    try:
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        print(f"Launch error: {e}")
        print("\nTroubleshooting:")
        print("1. Try: pip install -r requirements.txt")
        print("2. Create a virtual environment")
        print("3. Check Python version (3.8-3.11 recommended)")