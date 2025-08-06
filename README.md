LangChain RAG Chatbot with OpenAI GPT-4o Mini & ChromaDB
A powerful Retrieval-Augmented Generation (RAG) chatbot built with LangChain, featuring OpenAI's GPT-4o Mini and ChromaDB for intelligent document-based question answering. The application provides a modern Gradio web interface with comprehensive document management and chat capabilities.

🚀 Features
LangChain Integration: Built with LangChain for robust RAG implementation and LLM management
Advanced RAG System: Combines semantic document retrieval with generative AI for accurate, context-aware responses
Multiple Document Formats: Supports PDF, TXT, and Markdown files with intelligent chunking
Smart Text Splitting: Uses LangChain's RecursiveCharacterTextSplitter for optimal document processing
Vector Search: Powered by ChromaDB with OpenAI text-embedding-3-small for semantic document search
Modern UI: Clean and intuitive Gradio interface with organized tabs for different functions
Source Attribution: Shows reference documents with similarity scores and chunk information
Conversation Memory: Maintains contextual chat history for natural conversations
Performance Tracking: Built-in analytics for token usage, costs, and response times
Document Management: Upload, search, and manage your knowledge base with comprehensive controls
📋 Prerequisites
Python 3.8 or higher
OpenAI API key
Internet connection for API calls
🛠️ Installation
Clone or download the project files

Create a virtual environment (recommended):

bash

Copy
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
Install dependencies:

bash

Copy
pip install -r requirements.txt
Set up environment variables:

Create a .env file in the project root directory:

bash

Copy
# Copy the example file
copy env_example.txt .env
Edit the .env file and add your OpenAI API key:


Copy
OPENAI_API_KEY=your_actual_api_key_here
🏃 Quick Start
Run the application:

bash

Copy
python app.py
Open your browser and go to http://localhost:7860

Upload documents:

Go to the "📁 Upload" tab
Upload PDF, TXT, or MD files (supports multiple files)
Wait for processing confirmation
Start chatting:

Switch to the "💬 Chat" tab
Ask questions about your uploaded documents
View source references and similarity scores in responses
Toggle RAG on/off to compare responses with and without document context
📁 Project Structure
basic

Copy
├── app.py              # Main Gradio web application
├── chatbot.py          # LangChain-powered RAG chatbot implementation
├── rag_system.py       # ChromaDB integration and LangChain document processing
├── requirements.txt    # Python dependencies (LangChain ecosystem)
├── env_example.txt     # Environment variables template
├── README.md          # This file
└── chroma_db/         # ChromaDB persistent storage (created automatically)
🎯 Usage Guide
Chat Tab
Interactive Chat: Ask questions about your uploaded documents
Reference Documents: View source documents with similarity scores and chunk IDs
RAG Toggle: Enable/disable retrieval-augmented generation to compare responses
Response Metadata: See processing time, token usage, and costs for each response
Clear History: Reset the conversation for a fresh start
Upload Tab
Multi-File Upload: Upload multiple PDF, TXT, or MD files simultaneously
Intelligent Processing: Documents are automatically chunked using LangChain's text splitter
Processing Status: Real-time feedback on upload and processing status
Duplicate Detection: Automatic detection and handling of duplicate content
Search Tab
Semantic Search: Find relevant document chunks using natural language queries
Adjustable Parameters: Control number of results (1-10) and similarity threshold
Detailed Results: View document sources, similarity scores, and content previews
Direct Knowledge Base Access: Search without generating AI responses
Stats Tab
Knowledge Base Statistics: Document count, collection info, and storage details
Performance Dashboard: Comprehensive metrics including:
Total API calls and token usage
Response times and success rates
Cost tracking and averages
Model information and statistics
Performance Management: Reset statistics and refresh data
Settings Tab
System Information: View configuration details and model information
API Configuration: Check OpenAI API key status and model settings
Knowledge Base Management: Clear all documents when needed
Default Settings: View current similarity thresholds and processing parameters
⚙️ Configuration
Environment Variables
OPENAI_API_KEY: Your OpenAI API key (required)
Default Settings
LLM Model: gpt-4o-mini (LangChain ChatOpenAI)
Embedding Model: text-embedding-3-small (LangChain OpenAIEmbeddings)
Text Splitter: RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
Default Similarity Threshold: 0.3 (30%)
Max File Size: 50MB per file
Customization Options
You can modify the following in the code:

Text Splitting Parameters: Adjust chunk size and overlap in rag_system.py
Model Parameters: Change temperature, max_tokens in chatbot.py
Similarity Thresholds: Modify default values for document retrieval
UI Configuration: Customize Gradio interface themes and layouts
Performance Tracking: Adjust metrics collection and display
🔧 Troubleshooting
Common Issues
LangChain Import Errors:

Ensure you're using compatible LangChain versions
Install the complete requirements: pip install -r requirements.txt
Check for deprecated import warnings and update as needed
API Key Configuration:

Verify your .env file contains a valid OpenAI API key
Ensure the file is in the project root directory
Check API key permissions and quota limits
Document Processing Issues:

Verify file formats are supported (PDF, TXT, MD)
Check file size limits (50MB maximum)
Ensure sufficient disk space for ChromaDB storage
ChromaDB/Vector Database Errors:

Delete the chroma_db folder and restart the application
Check for permission issues in the project directory
Verify embedding model is accessible via OpenAI API
Performance Issues:

Large documents may take time to process initially
Monitor token usage and costs in the Stats tab
Use the performance dashboard to identify bottlenecks
Migration Notes
This version uses LangChain v0.2+ ecosystem
Deprecated imports are automatically detected and reported
Use the LangChain CLI migration tool if needed: langchain-cli migrate
📊 Performance Features
Real-time Metrics: Track API calls, tokens, and costs
Response Analytics: Monitor processing times and success rates
Cost Management: Detailed breakdown of usage costs per operation
Error Tracking: Comprehensive error logging and statistics
Token Optimization: Automatic callback tracking for usage monitoring
🤝 Contributing
Contributions are welcome! Please feel free to:

Submit issues for bugs or feature requests
Fork the repository and create pull requests
Improve documentation and examples
Share your customizations and use cases
📄 License
This project is open source and available under the MIT License.

🙏 Acknowledgments
LangChain for the comprehensive RAG framework and LLM abstractions
OpenAI for GPT-4o Mini and text-embedding-3-small models
ChromaDB for efficient vector database functionality
Gradio for the intuitive web interface framework
Pydantic for data validation and settings management
🔗 Related Resources
LangChain Documentation
OpenAI API Documentation
ChromaDB Documentation
Gradio Documentation