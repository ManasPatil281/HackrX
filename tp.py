from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any
import time
import os
import requests
import tempfile
import json
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
from datetime import datetime
import uuid
import asyncio
import pandas as pd
import csv
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import hashlib
import pickle
import json

from batch_processor import BatchProcessor
from doc_cache import DocumentCache
from docment_extracter import OCR_AVAILABLE, OFFICE_LIBS_AVAILABLE, detect_document_type, extract_document_content
from embeddings_fallback_manager import EmbeddingsFallbackManager
from llm_fallback_manager import LLMFallbackManager
from rate_limit_manager import RateLimitManager
from response_model import AnswerResponse, DebugRequest, QueryRequest
from logger import QueryLogger
from config import PORT, HOST, MISTRAL_API_KEY, HF_TOKEN


# Initialize document cache
document_cache = DocumentCache()


# Initialize FastAPI application with security documentation
app = FastAPI(
    title="RAG Backend API", 
    version="1.0.0",
    description="RAG Backend with Bearer Token Authentication and Hybrid Retrieval"
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

# Define token verification function
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the token provided in the Authorization header"""
    token = credentials.credentials
    # For demonstration purposes, we accept all tokens
    # In production, implement proper token validation here
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
security = HTTPBearer()


# Initialize rate limit manager
rate_limit_manager = RateLimitManager()


# Initialize query logger
query_logger = QueryLogger()


# Load environment variables from .env file
load_dotenv()

# Only suppress tokenizer parallelism warnings (still useful for HuggingFace)
os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'false')



# Initialize components with fallback managers
try:
    embeddings_manager = EmbeddingsFallbackManager()
    if embeddings_manager.embeddings:
        embeddings = embeddings_manager
        print("✅ Embeddings fallback manager initialized")
    else:
        # Original fallback logic
        try:
            embeddings = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-mpnet-base-v2",
                huggingfacehub_api_token=HF_TOKEN
            )
            print("✅ HuggingFace Endpoint Embeddings initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing endpoint embeddings: {e}")
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                print("✅ Fallback local embeddings initialized successfully")
            except Exception as e2:
                print(f"❌ All embedding methods failed: {e2}")
                embeddings = None
except Exception as e:
    print(f"❌ Error initializing embeddings: {e}")
    embeddings = None

try:
    llm_manager = LLMFallbackManager()
    if llm_manager.models:
        llm = llm_manager
        print("✅ LLM Fallback Manager initialized with Mistral models")
    else:
        # Fallback to basic Mistral if manager fails
        if MISTRAL_API_KEY:
            try:
                from langchain_mistralai import ChatMistralAI
                llm = ChatMistralAI(
                    model="mistral-large-2411",
                    mistral_api_key=MISTRAL_API_KEY,
                    temperature=0.2,
                    max_tokens=2000
                )
                print("✅ Basic Mistral LLM initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing basic Mistral LLM: {e}")
                llm = None
        else:
            print("❌ No MISTRAL_API_KEY found")
            llm = None
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

# Global variables to store the vector store and hybrid retriever
vector_store = None
hybrid_retriever = None
processed_documents = []
batch_processor = None
last_request_model = None  # Track last used model


# Update the process_documents_with_cache function to handle archive file detection
def process_documents_with_cache(documents: Union[List[str], str]) -> Dict[str, Any]:
    if isinstance(documents, str):
        documents = [documents]
    
    # Separate document URLs from text content
    doc_urls = []
    text_content = []
    unsupported_urls = []
    
    for doc in documents:
        if is_url(doc):
            doc_type = detect_document_type(doc)
            if doc_type == 'archive_unsupported':
                unsupported_urls.append(doc)
            elif doc_type != 'unknown':
                doc_urls.append(doc)
            else:
                print(f"⚠️ Unknown document type for {doc}, treating as text")
                text_content.append(doc)
        else:
            text_content.append(doc)
    
    # If there are unsupported archive URLs, raise an error
    if unsupported_urls:
        error_msg = f"The following URLs contain unsupported archive files: {', '.join(unsupported_urls)}. Archive files (.zip, .rar, .7z, .tar, .gz, .bz2) are not supported. Please extract the contents and provide direct links to individual documents."
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Check if we have complete cached data for all document URLs
    if doc_urls:
        cached_data = document_cache.get_cached_complete_data(doc_urls)
        if cached_data:
            print(f"⚡ Using complete cached data for {len(doc_urls)} documents!")
            # Add text content if any
            if text_content:
                cached_data["content"] += "\n\n" + "\n".join(text_content)
                # Recalculate content hash with text content
                cached_data["content_hash"] = document_cache.get_content_hash(cached_data["content"])
            
            return {
                "content": cached_data["content"],
                "content_hash": cached_data["content_hash"],
                "chunks": cached_data["chunks"],
                "vector_store": cached_data["vector_store"],
                "from_cache": True
            }
    
    # If not fully cached, process normally
    processed_content = []
    
    # Process document URLs
    for url in doc_urls:
        doc_content = extract_document_content(url)  # This uses individual document caching
        processed_content.append(doc_content)
    
    # Add text content
    processed_content.extend(text_content)
    
    final_content = "\n\n".join(processed_content)
    content_hash = document_cache.get_content_hash(final_content)
    
    return {
        "content": final_content,
        "content_hash": content_hash,
        "chunks": None,
        "vector_store": None,
        "from_cache": False,
        "pdf_urls": doc_urls  # Renamed to doc_urls but keeping same key for compatibility
    }

# Initialize FastAPI application with security documentation
app = FastAPI(
    title="Multi-Format RAG Backend API", 
    version="1.0.0",
    description="RAG Backend with Bearer Token Authentication and Hybrid Retrieval, supporting multiple document formats"
)

@app.get("/")
def root():
    return {
        "message": "Multi-Format RAG Backend with Mistral LLM and Caching",
        "status": "running",
        "features": [
            "Multiple Mistral LLM fallback",
            "Multiple embedding fallback (HuggingFace Local + Endpoint + Mistral)",
            "Hybrid retrieval (Vector + BM25 + MMR)",
            "Query logging and analytics",
            "Document and embedding caching",
            "Multi-format document support (PDF, DOC, PPT, Excel, Images, etc.)"
        ],
        "supported_formats": [
            "PDF (.pdf)",
            "Word Documents (.doc, .docx)",
            "PowerPoint (.ppt, .pptx)",
            "Excel (.xls, .xlsx, .xlsm)",
            "Images (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp)",
            "Text files (.txt, .md, .rtf)",
            "CSV (.csv)",
            "HTML (.html, .htm)",
            "JSON (.json)",
            "Plain text"
        ],
        "unsupported_formats": [
            "Archive files (.zip, .rar, .7z, .tar, .gz, .bz2) - Please extract and provide direct links to individual documents"
        ],
        "endpoints": {
            "health": "/health",
            "rag_status": "/rag-status",
            "run_query": "/hackrx/run",
            "debug_search": "/debug-search",
            "vector_stats": "/vector-stats",
            "llm_status": "/llm-status",
            "embeddings_status": "/embeddings-status",
            "query_stats": "/query-stats",
            "download_logs": "/download-logs",
            "cache_stats": "/cache-stats",
            "clear_cache": "/clear-cache"
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
        "llm_provider": "mistral",
        "llm_model": "mistral-large-2411",
        "llm_ready": llm is not None,
        "embedding_provider": "huggingface_local", 
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embeddings_ready": embeddings is not None,
        "vector_db": "faiss",
        "vector_store_ready": vector_store is not None,
        "chunk_size": 800,
        "chunk_overlap": 150,
        "framework": "langchain"
    }

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_query(request: QueryRequest, token: str = Depends(verify_token)):
    global vector_store, hybrid_retriever, processed_documents, batch_processor, last_request_model

    s = time.time()
    print("Request: ", request)
    # Check if required components are available
    if embeddings is None:
        raise HTTPException(status_code=500, detail="Embeddings not initialized")
    if llm is None:
        raise HTTPException(status_code=500, detail="LLM not initialized")
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Extract document links for logging
    document_links = []
    if isinstance(request.documents, list):
        document_links = [doc for doc in request.documents if is_url(doc)]
    elif isinstance(request.documents, str) and is_url(request.documents):
        document_links = [request.documents]
    
    # SMART MODEL ROTATION: Use different model for each request
    current_model_name = "unknown"
    if isinstance(llm, LLMFallbackManager):
        llm.rotate_to_next_available_model()
        current_model = llm.get_current_model_info()
        
        # Initialize last_request_model if None
        if last_request_model is None:
            last_request_model = current_model.get('name', 'unknown')
        
        if last_request_model == current_model.get('name'):
            llm.rotate_to_next_available_model()
            current_model = llm.get_current_model_info()
        
        current_model_name = current_model.get('name', 'unknown')
        last_request_model = current_model_name
        print(f"🎯 Using model for this request: {current_model_name} (provider: {current_model.get('provider', 'unknown')})")
    elif hasattr(llm, 'model'):
        current_model_name = llm.model
    
    print(f"🚀 Processing {len(request.questions)} questions with CACHING - Request: {request_id}")
    
    try:
        # Step 1: Process documents with caching (handles both text and URLs)
        doc_data = process_documents_with_cache(request.documents)
        document_content = doc_data["content"]
        content_hash = doc_data["content_hash"]
        document_type = "pdf" if any(is_url(doc) and 'pdf' in doc.lower() for doc in (request.documents if isinstance(request.documents, list) else [request.documents])) else "text"
        
        if document_content.strip():
            if doc_data["from_cache"]:
                # Use cached data
                print("⚡ Using fully cached data - no processing needed!")
                chunks = doc_data["chunks"]
                vector_store = doc_data["vector_store"]
                processed_documents = chunks
                
                # Initialize hybrid retriever
                hybrid_retriever = HybridRetriever(vector_store, processed_documents)
                print("✅ Loaded from cache and hybrid retriever created successfully")
            else:
                print("🔄 Cache miss - processing documents...")
                
                # Check if we have cached chunks
                cached_chunks = document_cache.get_cached_chunks(content_hash)
                cached_vector_store = document_cache.get_cached_vector_store(content_hash)
                
                if cached_chunks and cached_vector_store:
                    print("📁 Using cached chunks and vector store")
                    chunks = cached_chunks
                    vector_store = cached_vector_store
                    processed_documents = chunks
                else:
                    # Create documents
                    docs = [Document(page_content=document_content)]
                    
                    if cached_chunks:
                        print("📁 Using cached chunks")
                        chunks = cached_chunks
                    else:
                        # Split documents into chunks with optimized parameters
                        text_splitter_fast = RecursiveCharacterTextSplitter(
                            chunk_size=1500,  # Larger chunks for faster processing
                            chunk_overlap=250,  # Adequate overlap
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
                        chunks = text_splitter_fast.split_documents(docs)
                        print(f"Created {len(chunks)} chunks from documents")
                        
                        # Limit chunks for faster processing if too many
                        if len(chunks) > 400:
                            print(f"⚡ Limiting to 400 chunks (from {len(chunks)}) for maximum speed")
                            chunks = chunks[:400]
                        
                        # Cache the chunks
                        document_cache.cache_chunks(doc_data["content_hash"], chunks)
                    
                    # Create vector store if not cached
                    if cached_vector_store:
                        print("📁 Using cached vector store")
                        vector_store = cached_vector_store
                    else:
                        print("🔄 Creating new vector store...")
                        # Create vector store with fallback embeddings
                        if hasattr(embeddings, 'get_current_embedding'):
                            current_emb = embeddings.get_current_embedding()
                            if current_emb:
                                vector_store = FAISS.from_documents(chunks, current_emb)
                            else:
                                raise Exception("No current embedding available from fallback manager")
                        else:
                            # Direct embedding model (not a manager)
                            vector_store = FAISS.from_documents(chunks, embeddings)
                        
                        # Cache the vector store
                        document_cache.cache_vector_store(content_hash, vector_store)
                    
                    # Store processed documents for BM25
                    processed_documents = chunks
                
                # Cache complete data if we have PDF URLs and it's not from cache
                if not doc_data["from_cache"] and doc_data.get("pdf_urls"):
                    document_cache.cache_complete_data(
                        doc_data["pdf_urls"], 
                        document_content, 
                        chunks, 
                        vector_store
                    )
                
                # Initialize hybrid retriever
                hybrid_retriever = HybridRetriever(vector_store, processed_documents)
                print("✅ Vector store and hybrid retriever created successfully")
        
        # Initialize batch processor if needed with higher concurrency
        if batch_processor is None:
            batch_processor = BatchProcessor(llm, PROMPT, max_batch_size=len(request.questions))  # Allow all questions at once
            print(f"✅ Batch processor initialized for {len(request.questions)} concurrent questions")
        
        # Step 2: Answer ALL questions in FULL PARALLEL
        answers = []
        
        if vector_store is None:
            for i, question in enumerate(request.questions):
                answer = "No documents available for search"
                answers.append(answer)
                
                # Log failed question
                query_logger.log_query(
                    request_id=f"{request_id}-q{i+1}",
                    question=question,
                    document_links=document_links,
                    document_type=document_type,
                    answer=answer,
                    model_used=current_model_name,
                    processing_time=0,
                    chunks_retrieved=0,
                    success=False,
                    error_message="No documents provided"
                )
        else:
            print(f"🚀 Processing ALL {len(request.questions)} questions in FULL PARALLEL (no batching)")
            
            # Create semaphore for controlling max concurrent requests
            max_concurrent = min(len(request.questions), 10)  # Allow up to 10 concurrent questions
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_single_question_parallel(question, question_index):
                """Process a single question with full parallelization + dynamic API calls"""
                async with semaphore:
                    question_start_time = time.time()
                    chunks_retrieved = 0
                    
                    try:
                        print(f"🔍 Processing Q{question_index+1} in parallel: {question[:50]}...")
                        
                        # Retrieve relevant docs
                        if hybrid_retriever:
                            relevant_docs = hybrid_retriever.retrieve_relevant_docs(question, k=8)
                        else:
                            retriever = vector_store.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 8, "fetch_k": 16}
                            )
                            relevant_docs = retriever.get_relevant_documents(question)
                        
                        chunks_retrieved = len(relevant_docs)
                        print(f"   📊 Q{question_index+1}: Retrieved {chunks_retrieved} chunks")
                        
                        if relevant_docs:
                            # Build context text
                            context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

                            # Adapter to call LLM with messages
                            async def llm_direct(messages):
                                # Convert messages list to a single string
                                if isinstance(messages, list):
                                    prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
                                else:
                                    prompt = str(messages)

                                if hasattr(llm, 'invoke_with_fallback'):
                                    resp = await llm.invoke_with_fallback(prompt)
                                else:
                                    resp = await asyncio.to_thread(llm.invoke, prompt)

                                if hasattr(resp, "content"):
                                    return {"content": resp.content}
                                return {"content": str(resp)}


                            # Get final answer (supports multiple API calls)
                            answer = await run_llm_with_tools(llm_direct, question, context_text)
                            print(answer)
                            print(f"   ✅ Q{question_index+1} completed in {time.time() - question_start_time:.1f}s")
                            
                            # Log
                            question_time = time.time() - question_start_time
                            query_logger.log_query(
                                request_id=f"{request_id}-q{question_index+1}",
                                question=question,
                                document_links=document_links,
                                document_type=document_type,
                                answer=answer,
                                model_used=current_model_name,
                                processing_time=question_time,
                                chunks_retrieved=chunks_retrieved,
                                success=True,
                                error_message=""
                            )
                            return answer
                        
                        else:
                            answer = "No relevant information found for this question"
                            print(f"   ⚠️ Q{question_index+1}: No relevant docs found")
                            
                            question_time = time.time() - question_start_time
                            query_logger.log_query(
                                request_id=f"{request_id}-q{question_index+1}",
                                question=question,
                                document_links=document_links,
                                document_type=document_type,
                                answer=answer,
                                model_used=current_model_name,
                                processing_time=question_time,
                                chunks_retrieved=0,
                                success=False,
                                error_message="No relevant documents found"
                            )
                            return answer
                            
                    except Exception as e:
                        error_msg = f"Error processing question: {str(e)}"
                        print(f"❌ Error processing Q{question_index+1}: {str(e)}")
                        
                        question_time = time.time() - question_start_time
                        query_logger.log_query(
                            request_id=f"{request_id}-q{question_index+1}",
                            question=question,
                            document_links=document_links,
                            document_type=document_type,
                            answer="",
                            model_used=current_model_name,
                            processing_time=question_time,
                            chunks_retrieved=chunks_retrieved,
                            success=False,
                            error_message=str(e)
                        )
                        return error_msg
            
            # Process ALL questions in parallel at once
            parallel_start_time = time.time()
            tasks = [
                process_single_question_parallel(question, i) 
                for i, question in enumerate(request.questions)
            ]
            
            print(f"🚀 Launching {len(tasks)} parallel tasks...")
            parallel_answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            parallel_time = time.time() - parallel_start_time
            print(f"⚡ ALL {len(request.questions)} questions processed in {parallel_time:.2f} seconds!")
            
            # Collect results
            for i, result in enumerate(parallel_answers):
                if isinstance(result, Exception):
                    answers.append(f"Error processing question {i+1}: {str(result)}")
                else:
                    answers.append(result)
        
        processing_time = time.time() - start_time
        avg_time_per_question = processing_time / len(request.questions)
        
        print(f"🎉 CACHED RAG processing completed in {processing_time:.2f} seconds")
        print(f"📊 Average time per question: {avg_time_per_question:.2f} seconds")
        print(f"⚡ Speed improvement: {len(request.questions)}x parallelization + caching")
        print(f"🎯 Used model: {current_model_name}")
        print(f"📝 Logged {len(request.questions)} queries to CSV")
        print(f"🔍 Used {'hybrid' if hybrid_retriever else 'basic'} retrieval system")
        print(f"💾 Cache stats: {document_cache.get_cache_stats()}")
        
        print("Answers:", answers)
        print("time", time.time() - s)
        return AnswerResponse(answers=answers)
        
    except Exception as e:
        print(f"❌ Error in RAG processing: {str(e)}")
        
        # Log batch error
        for i, question in enumerate(request.questions):
            query_logger.log_query(
                request_id=f"{request_id}-q{i+1}",
                question=question,
                document_links=document_links,
                document_type="unknown",
                answer="",
                model_used=current_model_name,
                processing_time=0,
                chunks_retrieved=0,
                success=False,
                error_message=str(e)
            )
        
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

@app.get("/llm-status")
def llm_status():
    """Check LLM fallback manager status"""
    if isinstance(llm, LLMFallbackManager):
        model_info = llm.get_current_model_info()
        return {
            "fallback_manager_active": True,
            "current_model": model_info,
            "available_models": [{"name": m["name"], "provider": m["provider"]} for m in llm.models]
        }
    else:
        return {
            "fallback_manager_active": False,
            "llm_available": llm is not None
        }

@app.get("/embeddings-status")
def embeddings_status():
    """Check embeddings fallback manager status"""
    if isinstance(embeddings, EmbeddingsFallbackManager):
        return {
            "fallback_manager_active": True,
            "current_embedding": embeddings.embeddings[embeddings.current_index]["name"] if embeddings.embeddings else "none",
            "available_embeddings": [{"name": e["name"], "provider": e["provider"]} for e in embeddings.embeddings]
        }
    else:
        return {
            "fallback_manager_active": False,
            "embeddings_available": embeddings is not None
        }

@app.get("/query-stats")
def get_query_stats():
    """Get statistics from logged queries"""
    return query_logger.get_stats()

@app.get("/download-logs")
def download_logs():
    """Download the query logs CSV file"""
    if query_logger.log_file.exists():
        from fastapi.responses import FileResponse
        return FileResponse(
            path=query_logger.log_file,
            filename="query_logs.csv",
            media_type="text/csv"
        )
    else:
        raise HTTPException(status_code=404, detail="Log file not found")

@app.get("/cache-stats")
def get_cache_stats():
    """Get cache statistics"""
    return document_cache.get_cache_stats()

@app.post("/clear-cache")
async def clear_cache():
    """Clear all cached data"""
    success = document_cache.clear_cache()
    if success:
        return {"message": "Cache cleared successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear cache")


class HybridRetriever:
    """Enhanced retrieval system combining semantic and keyword search"""
    
    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        self.documents = documents
        self.setup_hybrid_retrieval()
    
    def setup_hybrid_retrieval(self):
        """Setup hybrid retrieval combining vector and BM25 search"""
        try:
            # Create BM25 retriever for keyword matching
            doc_texts = [doc.page_content for doc in self.documents if doc.page_content.strip()]
            
            if not doc_texts:
                print("⚠️ No valid document texts for BM25, using vector-only")
                self.bm25_retriever = None
                return
                
            self.bm25_retriever = BM25Retriever.from_texts(doc_texts)
            self.bm25_retriever.k = 10  # Increased for constitutional documents
            print("✅ BM25 retriever initialized with enhanced parameters")
        except Exception as e:
            print(f"⚠️ BM25 retriever failed, using vector-only: {e}")
            self.bm25_retriever = None
    
    def expand_query_terms(self, query: str) -> List[str]:
        """Expand query with related terms for any document type"""
        general_expansions = {
            # Legal and regulatory terms
            'provision': ['provision', 'clause', 'section', 'rule', 'requirement', 'stipulation'],
            'procedure': ['procedure', 'process', 'method', 'steps', 'approach', 'workflow'],
            'requirement': ['requirement', 'obligation', 'mandate', 'necessity', 'prerequisite'],
            'guideline': ['guideline', 'instruction', 'direction', 'recommendation', 'standard'],
            'policy': ['policy', 'rule', 'regulation', 'standard', 'protocol', 'guideline'],
            
            # Business and technical terms
            'implementation': ['implementation', 'execution', 'deployment', 'application', 'establishment'],
            'compliance': ['compliance', 'adherence', 'conformity', 'observance', 'fulfillment'],
            'responsibility': ['responsibility', 'duty', 'obligation', 'accountability', 'liability'],
            'authority': ['authority', 'power', 'jurisdiction', 'control', 'governance'],
            
            # Financial and organizational terms
            'budget': ['budget', 'financial', 'funding', 'allocation', 'expenditure', 'cost'],
            'organization': ['organization', 'structure', 'department', 'division', 'unit'],
            'management': ['management', 'administration', 'supervision', 'oversight', 'control'],
            
            # Time and schedule related
            'deadline': ['deadline', 'timeline', 'schedule', 'timeframe', 'due date'],
            'period': ['period', 'duration', 'timeframe', 'interval', 'term'],
            
            # Common document elements
            'definition': ['definition', 'meaning', 'interpretation', 'explanation', 'clarification'],
            'scope': ['scope', 'coverage', 'extent', 'range', 'application'],
            'objective': ['objective', 'goal', 'purpose', 'aim', 'target'],
            'criteria': ['criteria', 'standard', 'requirement', 'condition', 'specification']
        }
        
        expanded_terms = [query]
        query_lower = query.lower()
        
        # Look for exact matches and related terms
        for key, terms in general_expansions.items():
            if key in query_lower:
                expanded_terms.extend(terms)
        
        # Add number variations if present
        import re
        numbers = re.findall(r'\d+', query)
        for num in numbers:
            expanded_terms.extend([f'section {num}', f'chapter {num}', f'part {num}', f'clause {num}'])
        
        # Add common document structure variations
        if any(word in query_lower for word in ['section', 'chapter', 'part', 'clause']):
            expanded_terms.extend(['provision', 'subsection', 'paragraph', 'article'])
        
        return list(set(expanded_terms))
    
    def retrieve_relevant_docs(self, query: str, k: int = 12) -> List[Document]:
        """Enhanced retrieval for any document type with better coverage"""
        all_docs = []
        
        # Expand query terms for better document retrieval
        expanded_queries = self.expand_query_terms(query)
        
        print(f"🔍 Enhanced hybrid retrieval with {len(expanded_queries)} query variations")
        
        # 1. Vector-based retrieval with document focus
        for i, q in enumerate(expanded_queries[:4]):  # Use top 4 for document queries
            try:
                vector_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k//1.5, "fetch_k": k * 3}
                )
                vector_docs = vector_retriever.get_relevant_documents(q)
                all_docs.extend(vector_docs)
                print(f"   📊 Vector query {i+1}: {len(vector_docs)} docs")
            except Exception as e:
                print(f"⚠️ Vector retrieval failed for query {i+1}: {e}")
        
        # 2. BM25 keyword retrieval (excellent for exact terms and numbers)
        if self.bm25_retriever:
            try:
                for i, q in enumerate(expanded_queries[:3]):  # Use top 3 for BM25
                    bm25_docs = self.bm25_retriever.get_relevant_documents(q)
                    all_docs.extend(bm25_docs)
                    print(f"   🔤 BM25 query {i+1}: {len(bm25_docs)} docs")
            except Exception as e:
                print(f"⚠️ BM25 retrieval failed: {e}")
        
        # 3. MMR retrieval for diversity in document content
        try:
            mmr_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k//2, "fetch_k": k * 4, "lambda_mult": 0.8}  # Higher diversity
            )
            mmr_docs = mmr_retriever.get_relevant_documents(query)
            all_docs.extend(mmr_docs)
            print(f"   🎯 MMR retrieval: {len(mmr_docs)} docs")
        except Exception as e:
            print(f"⚠️ MMR retrieval failed: {e}")
        
        # Remove duplicates and score documents
        unique_docs = []
        seen_content = set()
        scored_docs = []
        
        for doc in all_docs:
            # Use first 300 chars for deduplication
            content_hash = hash(doc.page_content[:300])
            if content_hash not in seen_content:
                score = self.calculate_document_relevance_score(doc.page_content, query, expanded_queries)
                scored_docs.append((doc, score))
                seen_content.add(content_hash)
        
        # Sort by relevance score and return top docs
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents with higher threshold for complex queries
        final_k = min(k * 2, len(scored_docs))  # More documents for complex queries
        unique_docs = [doc for doc, score in scored_docs[:final_k]]
        
        print(f"✅ Enhanced hybrid retrieval: {len(unique_docs)} unique docs from {len(all_docs)} total")
        if scored_docs:
            top_scores = [score for _, score in scored_docs[:5]]
            print(f"📊 Top document relevance scores: {[round(s, 1) for s in top_scores]}")
        
        return unique_docs
    
    def calculate_document_relevance_score(self, content: str, original_query: str, expanded_queries: List[str]) -> float:
        """Calculate relevance score for any document type"""
        content_lower = content.lower()
        score = 0.0
        
        # Score based on original query terms (highest weight)
        original_terms = original_query.lower().split()
        for term in original_terms:
            if len(term) > 2:
                if term in content_lower:
                    # Higher score for numbers and specific identifiers
                    if term.isdigit() or any(char.isdigit() for char in term):
                        score += 5.0
                    else:
                        score += 3.0
        
        # Score based on expanded terms
        for query in expanded_queries[1:]:
            query_terms = query.lower().split()
            for term in query_terms:
                if len(term) > 2 and term in content_lower:
                    score += 2.0
        
        # Bonus for document structure terms
        structure_terms = [
            'section', 'chapter', 'part', 'clause', 'paragraph', 'article',
            'provision', 'subsection', 'schedule', 'appendix'
        ]
        for term in structure_terms:
            if term in content_lower:
                score += 2.0
        
        # Bonus for procedure and requirement indicators
        procedure_indicators = [
            'shall', 'must', 'will', 'should', 'may', 'required', 'mandatory',
            'procedure', 'process', 'guideline', 'instruction', 'requirement'
        ]
        for indicator in procedure_indicators:
            if indicator in content_lower:
                score += 1.0
        
        # Bonus for structured references with numbers
        import re
        structured_refs = re.findall(r'(section|chapter|part|clause|article)\s+\d+', content_lower)
        score += len(structured_refs) * 3.0
        
        return score

import json
import requests

# async def run_llm_with_tools(llm_func, question, context):
#     import json, requests

#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a helpful assistant. You can request web/API data by returning ONLY JSON like:\n"
#                 '{"action": "fetch_url", "url": "<link>"}\n'
#                 "Do not include any explanation or text when requesting data.\n"
#                 "Once you have all the data needed, return the final answer in plain text."
#             ),
#         },
#         {
#             "role": "user",
#             "content": f"Question: {question}\n\nContext:\n{context}"
#         }
#     ]

#     for step in range(10):  # up to 10 tool calls to prevent infinite loops
#         try:
#             # ✅ Ask the LLM what to do next
#             response = await llm_func(messages)
#             content = response["content"].strip()
#         except Exception as e:
#             return f"Error calling LLM: {e}"

#         # ✅ Try to parse as JSON to detect tool requests
#         try:
#             action_data = json.loads(content)
#             if (
#                 isinstance(action_data, dict)
#                 and action_data.get("action") == "fetch_url"
#                 and action_data.get("url")
#             ):
#                 fetch_url = action_data["url"]
#                 print(f"🌐 Step {step+1}: Fetching {fetch_url}")

#                 try:
#                     r = requests.get(fetch_url, timeout=10)
#                     r.raise_for_status()
#                     fetched_content = r.text
#                     print(fetched_content)
#                 except Exception as e:
#                     fetched_content = f"Error fetching {fetch_url}: {e}"

#                 # ✅ Give fetched data back to LLM as new context
#                 messages.append(
#                     {
#                         "role": "system",
#                         "content": f"Fetched content from {fetch_url}:\n{fetched_content}"
#                     }
#                 )
#                 continue  # Go to next loop iteration

#         except json.JSONDecodeError:
#             # Not JSON — means LLM is giving a plain-text final answer
#             return content

#         # If the output wasn’t JSON, assume it's the final answer
#         if not content.startswith("{"):
#             return content

#     return "Max tool iterations reached without final answer."

async def run_llm_with_tools(llm_func, question, context):
    """
    Runs an LLM to answer a question based on provided context.
    It first checks the context for explicit URLs, fetches their content,
    and then provides all information to the LLM for a final answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. You are provided with document context and potentially fetched web content. "
                "Answer the user's question based STRICTLY on this information. "
                "Do NOT perform any external searches. If the answer is not in the provided context, state that the information is not available."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}"
        }
    ]

    # New logic: Scan the initial context for explicit URLs
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    explicit_urls = re.findall(url_pattern, context)
    
    if explicit_urls:
        print(f"🔗 Found {len(explicit_urls)} explicit links in the documents. Fetching content...")
        
        for url in list(set(explicit_urls)): # Use a set to avoid fetching the same URL multiple times
            print(f"🌐 Fetching content from: {url}")
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                fetched_content = r.text
            except Exception as e:
                fetched_content = f"Error fetching {url}: {e}"
                
            messages.append(
                {
                    "role": "system",
                    "content": f"Fetched content from {url}:\n{fetched_content}"
                }
            )

    try:
        # Ask the LLM to generate a final answer based on the full, prepared context
        response = await llm_func(messages)
        content = response["content"].strip()
        return content
    except Exception as e:
        return f"Error calling LLM: {e}"
    
def create_optimized_context(relevant_docs, question, max_length=15000):
    """Create optimized context with enhanced relevance scoring for any document type"""
    if not relevant_docs:
        return ""
    
    # Enhanced query term analysis for general documents
    query_terms = question.lower().split()
    
    # General document structure terms get higher weight
    structure_terms = [
        'section', 'chapter', 'part', 'clause', 'paragraph', 'article', 'provision',
        'subsection', 'subpart', 'schedule', 'appendix', 'exhibit', 'attachment'
    ]
    
    # Procedural and process terms
    procedure_terms = [
        'procedure', 'process', 'steps', 'method', 'approach', 'implementation',
        'guidelines', 'instructions', 'requirements', 'criteria', 'standards'
    ]
    
    # Important action and legal terms
    action_terms = [
        'shall', 'must', 'will', 'should', 'may', 'can', 'required', 'mandatory',
        'optional', 'prohibited', 'allowed', 'permitted', 'authorized'
    ]
    
    # Score documents with improved algorithm for any document type
    scored_docs = []
    
    for doc in relevant_docs:
        content_lower = doc.page_content.lower()
        content = doc.page_content.strip()
        
        if not content:
            continue
        score = 0
        
        # Primary scoring: exact term matches
        for term in query_terms:
            if len(term) > 2:
                exact_matches = content_lower.count(term)
                # Higher weight for numbers and specific identifiers
                if term.isdigit() or any(char.isdigit() for char in term):
                    score += exact_matches * 4
                else:
                    score += exact_matches * 3
        
        # Document structure terminology bonus
        for term in structure_terms:
            if term in content_lower:
                score += 2
        
        # Procedure terminology bonus
        for term in procedure_terms:
            if term in content_lower:
                score += 2
        
        # Action/requirement terms bonus
        for term in action_terms:
            if term in content_lower:
                score += 1
        
        # Bonus for structured references (e.g., "Section 5.2", "Chapter 3")
        import re
        section_refs = re.findall(r'section\s+\d+', content_lower)
        chapter_refs = re.findall(r'chapter\s+\d+', content_lower)
        clause_refs = re.findall(r'clause\s*\(\w+\)', content_lower)
        page_refs = re.findall(r'page\s+\d+', content_lower)
        
        score += len(section_refs) * 3
        score += len(chapter_refs) * 3
        score += len(clause_refs) * 2
        score += len(page_refs) * 1
        
        # Bonus for numbered lists and structured content
        numbered_items = re.findall(r'\d+\.\s+|\(\d+\)|\d+\)', content)
        bullet_points = re.findall(r'•|\*\s+|-\s+', content)
        score += len(numbered_items) * 1
        score += len(bullet_points) * 0.5
        
        # Bonus for document formatting indicators
        formatting_indicators = ['provided that', 'subject to', 'in accordance with', 'as follows', 'including but not limited to']
        for indicator in formatting_indicators:
            if indicator in content_lower:
                score += 1
        
        scored_docs.append((doc, score, len(content)))
    
    # Sort by relevance score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Build context with clear section markers for any document type
    context_parts = []
    context_parts.append("=== RELEVANT DOCUMENT SECTIONS ===\n")
    
    current_length = 0
    section_num = 1
    
    for doc, score, content_length in scored_docs:
        content = doc.page_content.strip()
        
        # Enhanced section header for any document content
        section_header = f"\n--- SECTION {section_num} (Relevance: {score:.1f}) ---\n"
        total_addition = len(section_header) + len(content)
        
        if current_length + total_addition > max_length:
            # Try to fit a truncated version for important content
            remaining_space = max_length - current_length - len(section_header)
            if remaining_space > 500 and score > 3:  # Only include high-relevance truncated content
                truncated_content = content[:remaining_space] + "\n[...content continues in full document...]"
                context_parts.append(section_header)
                context_parts.append(truncated_content)
                current_length += len(section_header) + len(truncated_content)
            break
        
        context_parts.append(section_header)
        context_parts.append(content)
        current_length += total_addition
        section_num += 1
        
        # Stop if we have enough high-quality content
        if section_num > 8 and current_length > max_length * 0.8:
            break
    
    # Add document analysis footer
    context_parts.append(f"\n\n=== DOCUMENT ANALYSIS SUMMARY ===")
    context_parts.append(f"- Document sections analyzed: {section_num - 1}")
    context_parts.append(f"- Total content length: {current_length:,} characters")
    context_parts.append("- Focus on exact citations and document provisions")
    context_parts.append("- Distinguish between explicit content and implications")
    
    final_context = "\n".join(context_parts)
    
    # Debug output
    print(f"📄 Document context created: {len(final_context):,} chars from {section_num-1} sections")
    print(f"📊 Top relevance scores: {[round(score, 1) for _, score, _ in scored_docs[:3]]}")
    
    return final_context

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



@app.get("/debug-search")
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

@app.get("/dhruv.pdf")
def get_pdf():
    return FileResponse(r"D:\My_Space\Hackrx_Bajaj_Finserv\downloads\FinalRound4SubmissionPDF_20250809_091146.pdf", media_type="application/pdf")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Multi-Format RAG Backend Server...")
    print("📍 Server will be available at:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000")
    print("🔧 Features:")
    print("   - Multiple Mistral LLM fallback")
    print("   - Multiple embedding fallback (HuggingFace Local + Endpoint + Mistral)")
    print("   - Hybrid retrieval (Vector + BM25 + MMR)")
    print("   - Query logging and analytics")
    print("   - Intelligent document and embedding caching")
    print("📄 Supported Document Formats:")
    print("   - PDF, Word (DOC/DOCX), PowerPoint (PPT/PPTX)")
    print("   - Excel (XLS/XLSX), Images (JPG/PNG/etc.)")
    print("   - Text, CSV, HTML, JSON files")
    print("🚫 Unsupported Formats:")
    print("   - Archive files (.zip, .rar, .7z, .tar, .gz, .bz2)")
    print("   - Please extract archives and provide direct links to individual documents")
    print("🔧 OCR Support:")
    if OCR_AVAILABLE:
        print("   ✅ OCR enabled for image text extraction")
    else:
        print("   ⚠️ OCR not available - install: pip install easyocr pytesseract pillow")
    print("📊 Office Documents:")
    if OFFICE_LIBS_AVAILABLE:
        print("   ✅ Office document support enabled")
    else:
        print("   ⚠️ Office libs not available - install: pip install python-docx openpyxl python-pptx")
    
    uvicorn.run(app, host=HOST, port=PORT)