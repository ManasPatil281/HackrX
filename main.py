from fastapi import FastAPI, HTTPException,Depends,Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Union
import time
import os
import requests
import tempfile
from urllib.parse import urlparse
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Only suppress tokenizer parallelism warnings (still useful for HuggingFace)
os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'false')

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from langchain.prompts import PromptTemplate
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing dependencies:")
    print("pip install langchain langchain-community langchain-groq langchain-huggingface faiss-cpu pypdf requests")
    exit(1)

# Define request and response models
class DebugRequest(BaseModel):
    question: str

class QueryRequest(BaseModel):
    documents: Union[List[str], str]  # Allow both list of strings and single string
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# Initialize FastAPI application with security documentation
app = FastAPI(
    title="RAG Backend API", 
    version="1.0.0",
    description="RAG Backend with Bearer Token Authentication"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define security scheme for documentation  
security = HTTPBearer()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

# Custom prompt template for RAG
CUSTOM_PROMPT_TEMPLATE = """
You are a highly accurate assistant designed to answer customer questions based **only** on the provided policy context.

YOUR TASK:
- Search the entire context carefully and repeatedly to find the exact, complete answer.
- Pay special attention to terms like "coordination of benefits", "multiple insurance", "other insurance", "secondary claims", "remaining amount", "balance claim", or similar concepts.
- Only return facts found **explicitly** in the context; do not guess, assume, or infer.
- Use clear, friendly, and simple language.
- Answer in **1 to 3 sentences maximum**.
- **Do not** include line breaks, bullet points, formatting symbols, or additional commentary.
- Include specific conditions or sections in the policy in detail for the answer.
- If the answer is not found in the context, say: "This information is not available in the provided policy context."
- Your goal is to extract accurate, complete answers in natural, helpful language.
- Support your answer with the facts and evidence, mention it in the answer.

Context:
---------
{context}

Q: {question}
A:
"""

# Create the prompt template
PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# Initialize components with error handling
try:
    # Use HuggingFaceEndpointEmbeddings for remote API calls (no local TensorFlow)
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=HF_TOKEN
    )
    print("✅ HuggingFace Endpoint Embeddings initialized successfully")
except Exception as e:
    print(f"❌ Error initializing endpoint embeddings: {e}")
    # Fallback to local model (will use TensorFlow)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✅ Fallback local embeddings initialized successfully")
    except Exception as e2:
        print(f"❌ All embedding methods failed: {e2}")
        embeddings = None

try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    llm = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Smaller chunks for better granularity
    chunk_overlap=150,  # More overlap to preserve context
    separators=["\n\n", "\n", ". ", " ", ""]  # Better text splitting
)

# Helper function to check if string is a URL
def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

# Helper function to download and extract text from PDF
def extract_pdf_content(pdf_url: str) -> str:
    try:
        print(f"📥 Downloading PDF from: {pdf_url}")
        
        # Download the PDF
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        # Extract text using PyPDFLoader
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        # Combine all page content with better formatting
        content = ""
        for i, page in enumerate(pages):
            page_text = page.page_content.strip()
            if page_text:
                content += f"\n--- Page {i+1} ---\n{page_text}\n"
        
        # Clean up the content
        content = content.replace('\n\n\n', '\n\n')  # Remove excessive newlines
        content = content.replace('\t', ' ')  # Replace tabs with spaces
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        print(f"✅ PDF extracted successfully. Content length: {len(content)} characters")
        return content
        
    except Exception as e:
        print(f"❌ Error extracting PDF content: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF content: {str(e)}")

# Helper function to process documents (handles both text and URLs)
def process_documents(documents: Union[List[str], str]) -> str:
    if isinstance(documents, str):
        documents = [documents]
    
    processed_content = []
    
    for doc in documents:
        if is_url(doc):
            if doc.lower().endswith('.pdf') or 'pdf' in doc.lower():
                # Extract PDF content
                pdf_content = extract_pdf_content(doc)
                processed_content.append(pdf_content)
            else:
                raise HTTPException(status_code=400, detail=f"URL format not supported: {doc}")
        else:
            # Treat as text content
            processed_content.append(doc)
    
    return "\n".join(processed_content)

# Global variable to store the vector store
vector_store = None

@app.get("/")
def root():
    return {
        "message": "LangChain RAG Backend with Groq Llama + HuggingFace + FAISS",
        "status": "running",
        "supported_formats": ["text", "pdf_urls"],
        "endpoints": {
            "health": "/health",
            "rag_status": "/rag-status",
            "run_query": "/hackrx/run",
            "debug_search": "/debug-search",
            "vector_stats": "/vector-stats"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "embeddings_ready": embeddings is not None,
        "llm_ready": llm is not None,
        "vector_store_ready": vector_store is not None
    }

@app.get("/rag-status")
def rag_status():
    """Check RAG tool configuration and status."""
    return {
        "rag_tool_configured": True,
        "llm_provider": "groq",
        "llm_model": "llama-3.3-70b-versatile",
        "llm_ready": llm is not None,
        "embedding_provider": "huggingface_endpoint", 
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "embeddings_ready": embeddings is not None,
        "vector_db": "faiss",
        "vector_store_ready": vector_store is not None,
        "chunk_size": 800,
        "chunk_overlap": 150,
        "framework": "langchain"
    }

@app.post("/debug-search")
async def debug_search(request: DebugRequest):
    """Debug endpoint to see what chunks are retrieved for a question."""
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No vector store available")
    
    try:
        # Get retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6, "fetch_k": 12}
        )
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(request.question)
        
        # Format response
        retrieved_chunks = []
        for i, doc in enumerate(docs):
            retrieved_chunks.append({
                "chunk_id": i,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "full_length": len(doc.page_content)
            })
        
        return {
            "question": request.question,
            "total_chunks_retrieved": len(docs),
            "chunks": retrieved_chunks
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug search error: {str(e)}")

@app.get("/vector-stats")
async def vector_stats():
    """Get statistics about the vector store."""
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No vector store available")
    
    try:
        # Get basic stats
        total_vectors = vector_store.index.ntotal
        
        return {
            "total_vectors": total_vectors,
            "vector_dimension": vector_store.index.d if hasattr(vector_store.index, 'd') else "unknown",
            "index_type": str(type(vector_store.index)),
            "status": "ready"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "error"}
    
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    import sys
    print("\n" + "="*50, file=sys.stderr, flush=True)
    print("🔍 TOKEN VERIFICATION CALLED", file=sys.stderr, flush=True)
    print("="*50, file=sys.stderr, flush=True)
    
    received_token = credentials.credentials
    expected_token = os.getenv("AUTH_TOKEN")
    
    print(f"🔍 RECEIVED TOKEN: '{received_token}'", file=sys.stderr, flush=True)
    print(f"🔍 EXPECTED TOKEN: '{expected_token}'", file=sys.stderr, flush=True)
    print(f"🔍 LENGTHS - Received: {len(received_token)}, Expected: {len(expected_token) if expected_token else 0}", file=sys.stderr, flush=True)
    
    if not expected_token:
        print("❌ AUTH_TOKEN not set in environment", file=sys.stderr, flush=True)
        raise HTTPException(
            status_code=500, 
            detail="Server configuration error: AUTH_TOKEN not set."
        )

    # Try exact match first
    if received_token == expected_token:
        print("✅ EXACT TOKEN MATCH", file=sys.stderr, flush=True)
        return received_token
    
    # Try stripped match
    if received_token.strip() == expected_token.strip():
        print("✅ TOKEN MATCH AFTER STRIP", file=sys.stderr, flush=True)
        return received_token
    
    # If no match, show detailed comparison
    print("❌ TOKEN MISMATCH DETAILS:", file=sys.stderr, flush=True)
    print(f"  Received bytes: {received_token.encode()}", file=sys.stderr, flush=True)
    print(f"  Expected bytes: {expected_token.encode()}", file=sys.stderr, flush=True)
    
    raise HTTPException(
        status_code=403, 
        detail="Invalid or expired token."
    )

@app.get("/debug-token")
async def debug_token():
    """Debug endpoint to check token configuration."""
    expected_token = os.getenv("AUTH_TOKEN")
    return {
        "auth_token_set": expected_token is not None,
        "auth_token_length": len(expected_token) if expected_token else 0,
        "auth_token_first_10": expected_token[:10] if expected_token else None,
        "auth_token_last_10": expected_token[-10:] if expected_token else None,
        "full_token": expected_token  # Temporary for debugging
    }

@app.get("/test-auth")
async def test_auth(token: str = Depends(verify_token)):
    """Test endpoint to verify authentication is working."""
    return {"message": "Authentication successful!", "token_received": token[:10] + "..."}

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_query(request: QueryRequest, token: str = Depends(verify_token)):
    global vector_store
    
    # Check if required components are available
    if embeddings is None:
        raise HTTPException(status_code=500, detail="Embeddings not initialized")
    if llm is None:
        raise HTTPException(status_code=500, detail="LLM not initialized")
    
    try:
        start = time.time()
        
        # Step 1: Process documents (handles both text and URLs)
        document_content = process_documents(request.documents)
        
        if document_content.strip():
            # Create documents
            docs = [Document(page_content=document_content)]
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(docs)
            print(f"Created {len(chunks)} chunks from documents")
            
            # Create or update vector store
            if vector_store is None:
                print("Creating new vector store...")
                vector_store = FAISS.from_documents(chunks, embeddings)
                print("✅ Vector store created successfully")
            else:
                # Add new documents to existing vector store
                print("Adding documents to existing vector store...")
                vector_store.add_documents(chunks)
                print("✅ Documents added to vector store")
        
        # Step 2: Answer questions
        answers = []
        
        if vector_store is None:
            for question in request.questions:
                answers.append("No documents available for search")
        else:
            # Create retrieval QA chain with custom prompt and improved retrieval
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 6,  # Retrieve more chunks for better coverage
                        "fetch_k": 12  # Fetch more candidates before filtering
                    }
                ),
                return_source_documents=False,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            for question in request.questions:
                try:
                    print(f"Processing question: {question}")
                    
                    # Try multiple search variations for better retrieval
                    search_variations = [
                        question,
                        f"coordination of benefits {question}",
                        f"multiple insurance claims {question}",
                        "secondary insurance claim remaining amount",
                        "other insurance coordination benefits"
                    ]
                    
                    all_docs = []
                    for variation in search_variations:
                        retriever_temp = vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 3, "fetch_k": 6}
                        )
                        docs_temp = retriever_temp.get_relevant_documents(variation)
                        all_docs.extend(docs_temp)
                    
                    # Remove duplicates and get unique documents
                    unique_docs = []
                    seen_content = set()
                    for doc in all_docs:
                        if doc.page_content not in seen_content:
                            unique_docs.append(doc)
                            seen_content.add(doc.page_content)
                    
                    # Limit to top 6 unique documents
                    unique_docs = unique_docs[:6]
                    
                    print(f"Found {len(unique_docs)} unique relevant chunks")
                    
                    # Create a custom retriever that returns our enhanced results
                    if unique_docs:
                        # Manually create the context from retrieved docs
                        context = "\n\n".join([doc.page_content for doc in unique_docs])
                        
                        # Use the LLM directly with our custom prompt
                        formatted_prompt = PROMPT.format(context=context, question=question)
                        
                        llm_result = llm.invoke(formatted_prompt)
                        answer = llm_result.content if hasattr(llm_result, 'content') else str(llm_result)
                        answers.append(answer.strip())
                        print(f"✅ Answer generated for: {question}")
                    else:
                        # Fallback to original method
                        result = qa_chain.invoke({"query": question})
                        answer = result.get("result", "No answer found")
                        answers.append(answer.strip())
                        print(f"✅ Fallback answer generated for: {question}")
                        
                except Exception as e:
                    print(f"❌ Error answering question '{question}': {str(e)}")
                    answers.append(f"Error processing question: {str(e)}")
        
        processing_time = time.time() - start
        print(f"RAG processing completed in {processing_time:.2f} seconds")
        
        return AnswerResponse(answers=answers)
        
    except Exception as e:
        print(f"❌ Error in RAG processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Backend Server...")
    print("📍 Server will be available at:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000")
    print("📊 Endpoints:")
    print("   - GET  /            - Root endpoint")
    print("   - GET  /health      - Health check")
    print("   - GET  /rag-status  - RAG status")
    print("   - POST /hackrx/run  - Run RAG query")
    
    uvicorn.run(app, host=HOST, port=PORT)
