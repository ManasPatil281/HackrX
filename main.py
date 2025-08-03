from fastapi import FastAPI, HTTPException,Depends,Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


class BatchProcessor:
    """Handles parallel processing of questions with rate limiting"""
    
    def __init__(self, llm, prompt_template, max_batch_size=3):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_batch_size = max_batch_size
        self.semaphore = asyncio.Semaphore(max_batch_size)
    
    async def process_question(self, question, context):
        """Process a single question with context using the LLM"""
        try:
            # Format prompt with question and context
            formatted_prompt = self.prompt_template.format(
                question=question,
                context=context
            )
            
            # Use LLM to generate answer
            if hasattr(self.llm, 'invoke_with_fallback'):
                response = await self.llm.invoke_with_fallback(formatted_prompt)
            else:
                response = self.llm.invoke(formatted_prompt)
            
            # Extract answer from response
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            return answer
        except Exception as e:
            raise Exception(f"Error in batch processor: {str(e)}")


# Initialize FastAPI application with security documentation
app = FastAPI(
    title="RAG Backend API", 
    version="1.0.0",
    description="RAG Backend with Bearer Token Authentication and Hybrid Retrieval"
)

# Define request and response models
class DebugRequest(BaseModel):
    question: str

class QueryRequest(BaseModel):
    documents: Union[List[str], str]  # Allow both list of strings and single string
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

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

class RateLimitManager:
    """Manages API rate limits to prevent 429 errors"""
    
    def __init__(self):
        self.request_counts = {}
        self.last_reset = {}
        self.rate_limits = {
            'mistral': {
                'requests_per_minute': 100,
                'tokens_per_minute': 400000,
                'current_requests': 0,
                'current_tokens': 0
            },
            'groq': {
                'requests_per_minute': 200,
                'tokens_per_minute': 800000,
                'current_requests': 0,
                'current_tokens': 0
            }
        }
    
    def reset_if_needed(self, provider):
        now = time.time()
        if provider not in self.last_reset:
            self.last_reset[provider] = now
            return
        
        if now - self.last_reset[provider] >= 60:
            self.rate_limits[provider]['current_requests'] = 0
            self.rate_limits[provider]['current_tokens'] = 0
            self.last_reset[provider] = now
    
    def can_make_request(self, provider, estimated_tokens=1000):
        """Check if we can make a request without hitting limits"""
        self.reset_if_needed(provider)
        
        limits = self.rate_limits[provider]
        return (
            limits['current_requests'] < limits['requests_per_minute'] * 0.9 and
            limits['current_tokens'] + estimated_tokens < limits['tokens_per_minute'] * 0.9
        )
    
    def record_request(self, provider, tokens_used=1000):
        """Record a request to track usage"""
        self.reset_if_needed(provider)
        self.rate_limits[provider]['current_requests'] += 1
        self.rate_limits[provider]['current_tokens'] += tokens_used
    
    async def wait_if_needed(self, provider, estimated_tokens=1000):
        """Wait if we're approaching rate limits"""
        if not self.can_make_request(provider, estimated_tokens):
            wait_time = 60 - (time.time() - self.last_reset.get(provider, time.time()))
            if wait_time > 0:
                print(f"⏳ Rate limit approaching for {provider}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time + 1)

# Initialize rate limit manager
rate_limit_manager = RateLimitManager()

class QueryLogger:
    """Logs queries, documents, and answers to CSV for analysis"""
    
    def __init__(self, log_file="query_logs.csv"):
        self.log_file = Path(log_file)
        self.ensure_log_file_exists()
    
    def ensure_log_file_exists(self):
        if not self.log_file.exists():
            headers = [
                'timestamp', 'request_id', 'question', 'document_links',
                'document_type', 'answer', 'model_used', 'processing_time_seconds',
                'chunks_retrieved', 'success', 'error_message'
            ]
            
            with open(self.log_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
            print(f"✅ Created query log file: {self.log_file}")
    
    def log_query(self, request_id, question, document_links, document_type, answer, 
                  model_used, processing_time, chunks_retrieved=0, success=True, error_message=""):
        try:
            timestamp = datetime.now().isoformat()
            question_clean = question.replace('\n', ' ').replace('\r', ' ')[:500]
            answer_clean = answer.replace('\n', ' ').replace('\r', ' ')[:1000] if answer else ""
            links_str = "|".join(document_links) if isinstance(document_links, list) else str(document_links)
            
            row = [
                timestamp, request_id, question_clean, links_str, document_type,
                answer_clean, model_used, round(processing_time, 2), chunks_retrieved,
                success, error_message
            ]
            
            with open(self.log_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️ Error logging query: {e}")
    
    def get_stats(self):
        try:
            if not self.log_file.exists():
                return {"error": "No log file found"}
            
            df = pd.read_csv(self.log_file)
            return {
                "total_queries": len(df),
                "successful_queries": len(df[df['success'] == True]),
                "failed_queries": len(df[df['success'] == False]),
                "avg_processing_time": df['processing_time_seconds'].mean(),
                "success_rate": (len(df[df['success'] == True]) / len(df) * 100) if len(df) > 0 else 0
            }
        except Exception as e:
            return {"error": f"Error getting stats: {e}"}

# Initialize query logger
query_logger = QueryLogger()

class LLMFallbackManager:
    """Manages multiple LLM models with automatic fallback"""
    
    def __init__(self):
        self.models = []
        self.current_index = 0
        self.rate_limit_delays = {}
        self.rate_manager = RateLimitManager()  # Add rate limit manager
        self.request_count = 0  # Track total requests for rotation
        
        # Add Mistral models if available
        if MISTRAL_API_KEY:
            try:
                from langchain_mistralai import ChatMistralAI
                mistral_models = [
                    "mistral-large-2411",     # Latest large model
                    "mistral-large-2407",     # Previous large version
                    "mistral-large-2402",     # Earlier large version
                    "mistral-saba-2502",      # New specialized model
                    "mistral-medium",         # Medium capability
                    "mistral-small-2503",     # Latest small model
                    "mistral-small-2501",     # Previous small version
                    "mistral-small-2409",     # Earlier small version
                    "mistral-small-2402",     # Original small version
                    "ministral-8b-2410",     # Efficient 8B model
                    "ministral-3b-2410"      # Most efficient 3B model
                ]
                
                for model_name in mistral_models:
                    try:
                        mistral_model = ChatMistralAI(
                            model=model_name,
                            mistral_api_key=MISTRAL_API_KEY,
                            temperature=0.2,
                            max_tokens=2000
                        )
                        self.models.append({
                            "name": model_name,
                            "provider": "mistral",
                            "model": mistral_model
                        })
                        print(f"✅ Added Mistral model: {model_name}")
                    except Exception as e:
                        print(f"⚠️ Failed to initialize {model_name}: {e}")
                        continue
            except Exception as e:
                print(f"⚠️ Error initializing Mistral LLMs: {e}")
        
        print(f"✅ LLM Fallback Manager initialized with {len(self.models)} Mistral models")
    
    def get_current_model_info(self):
        if not self.models:
            return {"name": "none", "provider": "none"}
        return {
            "name": self.models[self.current_index]["name"],
            "provider": self.models[self.current_index]["provider"]
        }
    
    def rotate_to_next_available_model(self):
        """Intelligently rotate to next available model to distribute load"""
        if not self.models or len(self.models) == 1:
            return
        
        original_index = self.current_index
        attempts = 0
        
        while attempts < len(self.models):
            # Move to next model
            self.current_index = (self.current_index + 1) % len(self.models)
            model_info = self.models[self.current_index]
            model_key = f"{model_info['provider']}:{model_info['name']}"
            
            # Check if this model is available (not rate limited)
            if model_key not in self.rate_limit_delays:
                print(f"🔄 Rotated to model: {model_info['name']} (load balancing)")
                return
            
            # Check if rate limit has expired
            if time.time() > self.rate_limit_delays[model_key]:
                del self.rate_limit_delays[model_key]
                print(f"🔄 Rotated to model: {model_info['name']} (rate limit expired)")
                return
            
            attempts += 1
        
        # If all models are rate limited, stay on current
        print(f"⚠️ All models rate limited, staying on: {self.models[self.current_index]['name']}")
    
    def mark_rate_limited(self, index, delay_seconds=60):
        """Mark a model as rate-limited for a specified delay"""
        if not self.models or index >= len(self.models):
            return
        
        model_info = self.models[index]
        model_key = f"{model_info['provider']}:{model_info['name']}"
        self.rate_limit_delays[model_key] = time.time() + delay_seconds
        print(f"⚠️ Model {model_key} rate-limited for {delay_seconds} seconds")
        
        # Switch to next model
        self.current_index = (index + 1) % len(self.models)
    
    def invoke(self, prompt):
        if not self.models:
            raise ValueError("No LLM models available")
        
        errors = []
        
        # Try each model until one succeeds
        for _ in range(len(self.models)):
            try:
                model = self.models[self.current_index]["model"]
                result = model.invoke(prompt)
                return result
            except Exception as e:
                current_index = self.current_index
                error_message = str(e).lower()
                
                # Handle rate limit errors
                if "rate" in error_message and "limit" in error_message:
                    self.mark_rate_limited(current_index, delay_seconds=60)
                    errors.append(f"Rate limit for {self.models[current_index]['name']}")
                # Handle quota errors
                elif "quota" in error_message:
                    self.mark_rate_limited(current_index, delay_seconds=300)
                    errors.append(f"Quota exceeded for {self.models[current_index]['name']}")
                # Handle general errors
                else:
                    errors.append(f"Error with {self.models[current_index]['name']}: {str(e)}")
                    self.current_index = (self.current_index + 1) % len(self.models)
        
        # If all models failed
        error_msg = "; ".join(errors)
        raise RuntimeError(f"All LLM models failed: {error_msg}")
    
    async def invoke_with_fallback(self, prompt):
        """Async version of invoke with rate limiting and fallback handling"""
        if not self.models:
            raise ValueError("No LLM models available")
        
        # Estimate tokens for rate limiting
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimate
        
        errors = []
        
        # Try each model until one succeeds
        for attempt in range(len(self.models)):
            model_info = self.models[self.current_index]
            provider = model_info["provider"]
            
            try:
                # Check and wait for rate limits
                await self.rate_manager.wait_if_needed(provider, estimated_tokens)
                
                # Try to invoke the model
                result = await asyncio.to_thread(model_info["model"].invoke, prompt)
                
                # Record successful request
                self.rate_manager.record_request(provider, estimated_tokens)
                
                return result
                
            except Exception as e:
                error_message = str(e).lower()
                
                # Handle rate limit errors
                if "429" in error_message or "rate limit" in error_message or "quota" in error_message:
                    print(f"🚫 Rate limit hit for {model_info['name']}")
                    self.mark_rate_limited(self.current_index, delay_seconds=120)
                    errors.append(f"Rate limit for {model_info['name']}")
                else:
                    errors.append(f"Error with {model_info['name']}: {str(e)}")
                    
                # Move to next model
                self.current_index = (self.current_index + 1) % len(self.models)
        
        # All models failed
        error_msg = "; ".join(errors)
        raise RuntimeError(f"All LLM models failed: {error_msg}")

class EmbeddingsFallbackManager:
    """Manages fallback between different embedding providers"""
    
    def __init__(self):
        self.embeddings = []
        self.current_index = 0
        
        # HuggingFace local embeddings (PRIMARY)
        try:
            local_emb = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.embeddings.append({
                "name": "all-MiniLM-L6-v2",
                "provider": "huggingface_local",
                "embedding": local_emb
            })
            print("✅ HuggingFace local embeddings initialized as PRIMARY")
        except Exception as e:
            print(f"⚠️ HuggingFace local embeddings failed: {e}")
        
        # HuggingFace endpoint embeddings (fallback)
        if HF_TOKEN:
            try:
                endpoint_emb = HuggingFaceEndpointEmbeddings(
                    model="sentence-transformers/all-mpnet-base-v2",
                    huggingfacehub_api_token=HF_TOKEN
                )
                self.embeddings.append({
                    "name": "all-mpnet-base-v2-endpoint",
                    "provider": "huggingface_endpoint",
                    "embedding": endpoint_emb
                })
                print("✅ HuggingFace Endpoint embeddings initialized")
            except Exception as e:
                print(f"⚠️ HuggingFace Endpoint embeddings failed: {e}")
        
        # Mistral embeddings (last resort)
        if MISTRAL_API_KEY:
            try:
                from langchain_mistralai import MistralAIEmbeddings
                mistral_emb = MistralAIEmbeddings(
                    model="mistral-embed",
                    mistral_api_key=MISTRAL_API_KEY
                )
                self.embeddings.append({
                    "name": "mistral-embed",
                    "provider": "mistral",
                    "embedding": mistral_emb
                })
                print("✅ Mistral embeddings initialized")
            except Exception as e:
                print(f"⚠️ Mistral embeddings failed: {e}")
    
    def get_current_embedding(self):
        if not self.embeddings:
            return None
        return self.embeddings[self.current_index]["embedding"]
    
    def embed_documents(self, texts):
        for attempt in range(len(self.embeddings)):
            try:
                embedding = self.embeddings[self.current_index]["embedding"]
                return embedding.embed_documents(texts)
            except Exception as e:
                print(f"⚠️ Error with {self.embeddings[self.current_index]['name']}: {e}")
                self.current_index = (self.current_index + 1) % len(self.embeddings)
        raise Exception("All embedding models failed")
    
    def embed_query(self, text):
        for attempt in range(len(self.embeddings)):
            try:
                embedding = self.embeddings[self.current_index]["embedding"]
                return embedding.embed_query(text)
            except Exception as e:
                print(f"⚠️ Error with {self.embeddings[self.current_index]['name']}: {e}")
                self.current_index = (self.current_index + 1) % len(self.embeddings)
        raise Exception("All embedding models failed")

class QueryLogger:
    """Logs queries, documents, and answers to CSV for analysis"""
    
    def __init__(self, log_file="query_logs.csv"):
        self.log_file = Path(log_file)
        self.ensure_log_file_exists()
    
    def ensure_log_file_exists(self):
        if not self.log_file.exists():
            headers = [
                'timestamp', 'request_id', 'question', 'document_links',
                'document_type', 'answer', 'model_used', 'processing_time_seconds',
                'chunks_retrieved', 'success', 'error_message'
            ]
            
            with open(self.log_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
            print(f"✅ Created query log file: {self.log_file}")
    
    def log_query(self, request_id, question, document_links, document_type, answer, 
                  model_used, processing_time, chunks_retrieved=0, success=True, error_message=""):
        try:
            timestamp = datetime.now().isoformat()
            question_clean = question.replace('\n', ' ').replace('\r', ' ')[:500]
            answer_clean = answer.replace('\n', ' ').replace('\r', ' ')[:1000] if answer else ""
            links_str = "|".join(document_links) if isinstance(document_links, list) else str(document_links)
            
            row = [
                timestamp, request_id, question_clean, links_str, document_type,
                answer_clean, model_used, round(processing_time, 2), chunks_retrieved,
                success, error_message
            ]
            
            with open(self.log_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️ Error logging query: {e}")
    
    def get_stats(self):
        try:
            if not self.log_file.exists():
                return {"error": "No log file found"}
            
            df = pd.read_csv(self.log_file)
            return {
                "total_queries": len(df),
                "successful_queries": len(df[df['success'] == True]),
                "failed_queries": len(df[df['success'] == False]),
                "avg_processing_time": df['processing_time_seconds'].mean(),
                "success_rate": (len(df[df['success'] == True]) / len(df) * 100) if len(df) > 0 else 0
            }
        except Exception as e:
            return {"error": f"Error getting stats: {e}"}

# Load environment variables from .env file
load_dotenv()

# Only suppress tokenizer parallelism warnings (still useful for HuggingFace)
os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'false')

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

# Enhanced prompt template for any document type
CUSTOM_PROMPT_TEMPLATE = """
You are an expert document analyst providing precise, authoritative answers based STRICTLY on the document context provided.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

CRITICAL INSTRUCTIONS FOR DOCUMENT ANALYSIS:
1. **CONTENT-SPECIFIC FOCUS**: When asked about specific sections, clauses, or provisions, locate and cite the EXACT text
2. **PROCEDURAL PRECISION**: For procedural questions, provide step-by-step processes as outlined in the document
3. **ACCURATE TERMINOLOGY**: Use precise terminology from the document and maintain accuracy
4. **CITE EXACT SOURCES**: Always reference specific sections, clauses, page numbers, or document parts
5. **DISTINGUISH WHAT'S AVAILABLE**: Clearly state when specific information is not in the provided context

RESPONSE STRUCTURE FOR DOCUMENT ANALYSIS:
**DIRECT ANSWER**: [Specific, precise response to the question]

**SUPPORTING EVIDENCE**: 
- Section/Page X: [Exact text or precise summary with citations]
- [Additional relevant provisions with exact references]

**PROCEDURE/PROCESS** (if applicable):
- Step 1: [As specified in the document]
- Step 2: [As specified in the document]
- [Continue as needed]

**ANALYSIS**:
[Analysis based on the document's content and context]

**LIMITATIONS OF AVAILABLE INFORMATION**:
[What aspects cannot be answered from the provided context]

IMPORTANT RULES:
- Quote exact text from the document when available
- Use proper citation format (Section X, Page Y, Clause Z, etc.)
- If specific information is not in the context, state this clearly
- Distinguish between what is explicitly stated vs. what is implied
- For complex questions, break down into logical components
- Maintain objectivity and stick to documented facts

ANSWER:"""


# Create the enhanced prompt template
PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

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

# Global variables to store the vector store and hybrid retriever
vector_store = None
hybrid_retriever = None
processed_documents = []
batch_processor = None
last_request_model = None  # Track last used model

@app.get("/")
def root():
    return {
        "message": "LangChain RAG Backend with Mistral LLM",
        "status": "running",
        "features": [
            "Multiple Mistral LLM fallback",
            "Multiple embedding fallback (HuggingFace Local + Endpoint + Mistral)",
            "Hybrid retrieval (Vector + BM25 + MMR)",
            "Query logging and analytics"
        ],
        "supported_formats": ["text", "pdf_urls"],
        "endpoints": {
            "health": "/health",
            "rag_status": "/rag-status",
            "run_query": "/hackrx/run",
            "debug_search": "/debug-search",
            "vector_stats": "/vector-stats",
            "llm_status": "/llm-status",
            "embeddings_status": "/embeddings-status",
            "query_stats": "/query-stats",
            "download_logs": "/download-logs"
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
                    search_kwargs={"k": k//2, "fetch_k": k * 3}
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

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_query(request: QueryRequest, token: str = Depends(verify_token)):
    global vector_store, hybrid_retriever, processed_documents, batch_processor, last_request_model
    
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
    
    print(f"🚀 Processing {len(request.questions)} questions in FULL PARALLEL MODE - Request: {request_id}")
    
    try:
        # Step 1: Process documents (handles both text and URLs)
        document_content = process_documents(request.documents)
        document_type = "pdf" if any(is_url(doc) and 'pdf' in doc.lower() for doc in (request.documents if isinstance(request.documents, list) else [request.documents])) else "text"
        
        if document_content.strip():
            # Create documents
            docs = [Document(page_content=document_content)]
            
            # Split documents into chunks with optimized parameters for speed
            text_splitter_fast = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Even larger chunks for faster processing
                chunk_overlap=250,  # Adequate overlap
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter_fast.split_documents(docs)
            print(f"Created {len(chunks)} chunks from documents (optimized for speed)")
            
            # Limit chunks for faster processing if too many
            if len(chunks) > 400:  # Further reduced limit
                print(f"⚡ Limiting to 400 chunks (from {len(chunks)}) for maximum speed")
                chunks = chunks[:400]
            
            # Create or update vector store with fallback embeddings
            if vector_store is None:
                print("Creating new vector store...")
                # Fix: Use the proper method to get embedding from manager
                if hasattr(embeddings, 'get_current_embedding'):
                    current_emb = embeddings.get_current_embedding()
                    if current_emb:
                        vector_store = FAISS.from_documents(chunks, current_emb)
                    else:
                        raise Exception("No current embedding available from fallback manager")
                else:
                    # Direct embedding model (not a manager)
                    vector_store = FAISS.from_documents(chunks, embeddings)
                
                # Store processed documents for BM25
                processed_documents = chunks
                
                # Initialize hybrid retriever
                hybrid_retriever = HybridRetriever(vector_store, processed_documents)
                print("✅ Vector store and hybrid retriever created successfully")
            else:
                # Add new documents to existing vector store
                print("Adding documents to existing vector store...")
                # Fix: Use the proper method for adding documents
                if hasattr(embeddings, 'get_current_embedding'):
                    current_emb = embeddings.get_current_embedding()
                    if current_emb:
                        # Create temporary vector store for new chunks
                        temp_vector_store = FAISS.from_documents(chunks, current_emb)
                        # Merge with existing vector store
                        vector_store.merge_from(temp_vector_store)
                    else:
                        raise Exception("No current embedding available from fallback manager")
                else:
                    # Direct embedding model
                    vector_store.add_documents(chunks)
                
                processed_documents.extend(chunks)
                
                # Reinitialize hybrid retriever with updated documents
                hybrid_retriever = HybridRetriever(vector_store, processed_documents)
                print("✅ Documents added to vector store and hybrid retriever updated")
        
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
                """Process a single question with full parallelization"""
                async with semaphore:
                    question_start_time = time.time()
                    chunks_retrieved = 0
                    
                    try:
                        print(f"🔍 Processing Q{question_index+1} in parallel: {question[:50]}...")
                        
                        # Use hybrid retriever with optimized parameters for speed
                        if hybrid_retriever:
                            relevant_docs = hybrid_retriever.retrieve_relevant_docs(question, k=5)  # Reduced to 5 for speed
                        else:
                            # Enhanced fallback retrieval with reduced parameters
                            retriever = vector_store.as_retriever(
                                search_type="similarity",  # Faster than MMR
                                search_kwargs={"k": 5, "fetch_k": 10}  # Further reduced for speed
                            )
                            relevant_docs = retriever.get_relevant_documents(question)
                        
                        chunks_retrieved = len(relevant_docs)
                        print(f"   📊 Q{question_index+1}: Retrieved {chunks_retrieved} chunks")
                        
                        if relevant_docs:
                            # Create optimized context with reduced length for speed
                            context = create_optimized_context(relevant_docs, question, max_length=6000)  # Further reduced
                            
                            # Use LLM directly for maximum speed (no additional batch processing)
                            formatted_prompt = PROMPT.format(context=context, question=question)
                            
                            # Use LLM with fallback
                            if hasattr(llm, 'invoke_with_fallback'):
                                response = await llm.invoke_with_fallback(formatted_prompt)
                            else:
                                # Use asyncio.to_thread for non-async LLM
                                response = await asyncio.to_thread(llm.invoke, formatted_prompt)
                            

                            # Extract answer
                            if hasattr(response, 'content'):
                                answer = response.content
                            else:
                                answer = str(response)
                            

                            print(f"   ✅ Q{question_index+1} completed in {time.time() - question_start_time:.1f}s")
                            
                            # Log successful question
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
                            
                            # Log question with no relevant docs
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
                        
                        # Log failed question
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
        
        print(f"🎉 FULL PARALLEL RAG processing completed in {processing_time:.2f} seconds")
        print(f"📊 Average time per question: {avg_time_per_question:.2f} seconds")
        print(f"⚡ Speed improvement: {len(request.questions)}x parallelization")
        print(f"🎯 Used model: {current_model_name}")
        print(f"📝 Logged {len(request.questions)} queries to CSV")
        print(f"🔍 Used {'hybrid' if hybrid_retriever else 'basic'} retrieval system")
        
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

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Backend Server with Mistral LLM...")
    print("📍 Server will be available at:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000")
    print("🔧 Features:")
    print("   - Multiple Mistral LLM fallback")
    print("   - Multiple embedding fallback (HuggingFace Local + Endpoint + Mistral)")
    print("   - Hybrid retrieval (Vector + BM25 + MMR)")
    print("   - Query logging and analytics")
    print("📊 Endpoints:")
    print("   - GET  /            - Root endpoint")
    print("   - GET  /health      - Health check")
    print("   - GET  /rag-status  - RAG status")
    print("   - POST /hackrx/run  - Run RAG query")
    print("   - GET  /llm-status  - LLM fallback status")
    print("   - GET  /embeddings-status - Embeddings fallback status")
    print("   - GET  /query-stats - Query statistics")
    print("   - GET  /download-logs - Download CSV logs")
    
    uvicorn.run(app, host=HOST, port=PORT)
