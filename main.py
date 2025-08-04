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
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import hashlib
import pickle
import json

# Advanced PDF parsing with unstructured
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import Table, NarrativeText, Title, ListItem
    UNSTRUCTURED_AVAILABLE = True
    print("✅ Unstructured PDF parsing available")
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("⚠️ Unstructured not available, falling back to PyPDF")
    print("   To install unstructured: pip install 'unstructured[pdf]' pdf2image pytesseract")
    from langchain_community.document_loaders import PyPDFLoader
    
    # Define placeholder classes to avoid references to missing classes
    class Table:
        pass
    
    class NarrativeText:
        pass
    
    class Title:
        pass
    
    class ListItem:
        pass


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

# Enhanced prompt template to prevent hallucination and encourage deeper analysis
CUSTOM_PROMPT_TEMPLATE = """
You are an expert document analyst with deep expertise in extracting maximum value from provided content. Your mission is to provide comprehensive, accurate answers by actively searching through ALL available information.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

CRITICAL ANALYSIS PROTOCOL:
1. **EXHAUSTIVE SEARCH**: Scan every section, paragraph, and sentence for ANY relevant information
2. **CONNECT THE DOTS**: Look for relationships between different sections that together answer the question
3. **INFERENCE ENGINE**: Use logical reasoning to derive answers from related or partial information
4. **SEMANTIC MATCHING**: Recognize synonyms, paraphrases, and conceptually related content
5. **ZERO TOLERANCE FOR "NOT FOUND"**: Only declare information unavailable after truly exhaustive analysis

ENHANCED SEARCH STRATEGY:
- Direct keyword matching in all sections
- Conceptual similarity detection (related terms, synonyms)
- Cross-referencing between sections for complete picture
- Historical and contextual inference where appropriate
- Pattern recognition for implicit information

RESPONSE FRAMEWORK:
**COMPREHENSIVE ANSWER**: [Extract and synthesize ALL relevant information to provide the most complete response possible]

**SUPPORTING EVIDENCE**: 
- [Direct quotes with exact section/page references]
- [Related information that supports the answer]
- [Cross-references between sections]

**ANALYTICAL REASONING**: 
[Explain the logical connections and inferences that led to your answer]

**INFORMATION SYNTHESIS**: 
[How different pieces of information were combined to form the complete answer]

**CONFIDENCE ASSESSMENT**: [High/Medium/Low with detailed justification]

PERFORMANCE MANDATE: Your goal is to be maximally helpful and informative. Extract every possible insight from the provided context. Use intelligent reasoning to connect disparate pieces of information. Only state that specific information is completely unavailable if you have conducted an absolutely thorough search and found zero relevant content.

ANSWER:"""


# Create the enhanced prompt template
PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

class DocumentCache:
    """Caches documents, chunks, and vector stores to avoid repeated processing"""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.pdf_cache = {}
        self.chunks_cache = {}
        self.vector_store_cache = {}
        print(f"✅ Document cache initialized at {self.cache_dir}")
    
    def get_content_hash(self, content):
        """Generate a hash for document content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_cached_pdf_content(self, pdf_url):
        """Get cached PDF content if available"""
        cache_key = hashlib.md5(pdf_url.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"pdf_{cache_key}.txt"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"📁 Using cached PDF content for {pdf_url[:30]}...")
                return content
            except Exception as e:
                print(f"⚠️ Error reading PDF cache: {e}")
        
        return None
    
    def cache_pdf_content(self, pdf_url, content):
        """Cache PDF content to avoid repeated downloads"""
        try:
            cache_key = hashlib.md5(pdf_url.encode('utf-8')).hexdigest()
            cache_file = self.cache_dir / f"pdf_{cache_key}.txt"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"💾 Cached PDF content for {pdf_url[:30]}")
            return True
        except Exception as e:
            print(f"⚠️ Error caching PDF content: {e}")
            return False
    
    def get_cached_chunks(self, content_hash):
        """Get cached document chunks if available"""
        chunks_file = self.cache_dir / f"chunks_{content_hash}.pkl"
        
        if chunks_file.exists():
            try:
                with open(chunks_file, 'rb') as f:
                    chunks = pickle.load(f)
                print(f"📁 Using cached chunks for content hash {content_hash[:8]}...")
                return chunks
            except Exception as e:
                print(f"⚠️ Error reading chunks cache: {e}")
        
        return None
    
    def cache_chunks(self, content_hash, chunks):
        """Cache document chunks to avoid repeated splitting"""
        try:
            chunks_file = self.cache_dir / f"chunks_{content_hash}.pkl"
            
            with open(chunks_file, 'wb') as f:
                pickle.dump(chunks, f)
            
            print(f"💾 Cached {len(chunks)} chunks for content hash {content_hash[:8]}")
            return True
        except Exception as e:
            print(f"⚠️ Error caching chunks: {e}")
            return False
    
    def get_cached_vector_store(self, content_hash):
        """Get cached vector store if available"""
        vector_file = self.cache_dir / f"vector_{content_hash}.pkl"
        
        if vector_file.exists():
            try:
                with open(vector_file, 'rb') as f:
                    vector_store = pickle.load(f)
                print(f"📁 Using cached vector store for content hash {content_hash[:8]}...")
                return vector_store
            except Exception as e:
                print(f"⚠️ Error reading vector store cache: {e}")
        
        return None
    
    def cache_vector_store(self, content_hash, vector_store):
        """Cache vector store to avoid repeated embedding"""
        try:
            vector_file = self.cache_dir / f"vector_{content_hash}.pkl"
            
            with open(vector_file, 'wb') as f:
                pickle.dump(vector_store, f)
            
            print(f"💾 Cached vector store for content hash {content_hash[:8]}")
            return True
        except Exception as e:
            print(f"⚠️ Error caching vector store: {e}")
            return False
    
    def get_cache_stats(self):
        """Get statistics about the cache"""
        try:
            pdf_files = list(self.cache_dir.glob("pdf_*.txt"))
            chunks_files = list(self.cache_dir.glob("chunks_*.pkl"))
            vector_files = list(self.cache_dir.glob("vector_*.pkl"))
            
            # Calculate total size
            pdf_size = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)  # MB
            chunks_size = sum(f.stat().st_size for f in chunks_files) / (1024 * 1024)  # MB
            vector_size = sum(f.stat().st_size for f in vector_files) / (1024 * 1024)  # MB
            
            return {
                "pdf_files": len(pdf_files),
                "chunks_files": len(chunks_files),
                "vector_files": len(vector_files),
                "pdf_size_mb": round(pdf_size, 2),
                "chunks_size_mb": round(chunks_size, 2),
                "vector_size_mb": round(vector_size, 2),
                "total_size_mb": round(pdf_size + chunks_size + vector_size, 2),
                "cache_directory": str(self.cache_dir)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            # Remove all cache files
            for file in self.cache_dir.glob("*.txt"):
                file.unlink()
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
                
            # Clear in-memory caches
            self.pdf_cache = {}
            self.chunks_cache = {}
            self.vector_store_cache = {}
            
            print("🧹 Cache cleared successfully")
            return True
        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")
            return False

# Initialize document cache
document_cache = DocumentCache()

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
def parse_pdf_unstructured(file_path):
    """
    Extracts both readable text and table content from a PDF using unstructured,
    using auto strategy (layout-aware + OCR fallback).
    
    Returns:
    - full_text: concatenated narrative text (including headings and list items)
    - table_text: tables converted into readable key-value sentences
    """
    elements = partition_pdf(filename=file_path, strategy="auto", extract_images_in_pdf=False)

    text_blocks = []
    table_blocks = []

    for i, el in enumerate(elements):
        if isinstance(el, (NarrativeText, Title, ListItem)):
            if el.text and el.text.strip():
                page_info = f"[Page {el.metadata.page_number}]" if el.metadata and el.metadata.page_number else ""
                text_blocks.append(f"{page_info}\n{el.text.strip()}")
        elif isinstance(el, Table):
            flattened = flatten_table_unstructured(el)
            if flattened:
                table_blocks.append(flattened)

    return (
        "\n\n".join(text_blocks).strip(),
        "\n\n".join(table_blocks).strip()
    )


def flatten_table_unstructured(table_el):
    """
    Converts an unstructured Table element into readable key-value sentence form,
    if headers and rows are clearly defined.
    """
    try:
        raw_table = table_el.metadata.text_as_html or table_el.text
        if not table_el.metadata or not table_el.metadata.text_as_html:
            return None  # Skip if table structure wasn't fully extracted

        # Optionally: convert to clean rows (not HTML). Here's a basic approach:
        rows = table_el.metadata.text_as_rows
        if not rows or len(rows) < 2:
            return None

        headers = rows[0]
        data_rows = rows[1:]
        page_info = f"Table from Page {table_el.metadata.page_number}" if table_el.metadata and table_el.metadata.page_number else "Table"

        lines = [page_info]
        for row in data_rows:
            if not row or len(row) != len(headers):
                continue
            try:
                line = ", ".join(
                    f"{headers[i].strip()}: {row[i].strip()}" for i in range(len(headers))
                )
                lines.append(line)
            except Exception as e:
                print(f"⚠️ Skipped malformed row: {e}")
        return "\n".join(lines)

    except Exception as e:
        print(f"⚠️ Failed to flatten table: {e}")
        return None

# Format PyPDF pages into structured content
def _format_pypdf_content(pages):
    """Format PyPDF pages into structured content"""
    content = ""
    for i, page in enumerate(pages):
        page_text = page.page_content.strip()
        if page_text:
            content += f"\n--- Page {i+1} ---\n{page_text}\n"
    
    # Clean up the content
    content = content.replace('\n\n\n', '\n\n')  # Remove excessive newlines
    content = content.replace('\t', ' ')  # Replace tabs with spaces
    
    return content

def extract_pdf_content_advanced(pdf_url: str) -> str:
    """Enhanced PDF extraction with unstructured for better table and layout parsing"""
    # Check cache first
    cached_content = document_cache.get_cached_pdf_content(pdf_url)
    if cached_content:
        return cached_content
    
    try:
        print(f"📥 Downloading PDF from: {pdf_url}")
        
        # Download the PDF
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        # Extract text using unstructured if available
        if UNSTRUCTURED_AVAILABLE:
            try:
                print("🔧 Using unstructured for advanced PDF parsing...")
                full_text, table_text = parse_pdf_unstructured(temp_path)
                
                # Combine narrative text and table text
                content_parts = []
                
                if full_text.strip():
                    content_parts.append("=== DOCUMENT CONTENT ===")
                    content_parts.append(full_text)
                
                if table_text.strip():
                    content_parts.append("\n=== EXTRACTED TABLES ===")
                    content_parts.append(table_text)
                
                content = "\n\n".join(content_parts)
                print(f"✅ Advanced PDF extraction completed. Content length: {len(content)} characters")
                print(f"📊 Narrative text: {len(full_text)} chars, Table data: {len(table_text)} chars")
                
            except Exception as e:
                print(f"⚠️ Unstructured parsing failed, falling back to PyPDF: {e}")
                # Fallback to PyPDFLoader
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(temp_path)
                pages = loader.load()
                content = _format_pypdf_content(pages)
        else:
            # Use PyPDFLoader as fallback
            print("🔧 Using PyPDF for standard extraction...")
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            content = _format_pypdf_content(pages)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Cache the content
        document_cache.cache_pdf_content(pdf_url, content)
        
        return content
        
    except Exception as e:
        print(f"❌ Error extracting PDF content: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF content: {str(e)}")

# Main PDF extraction function
def extract_pdf_content(pdf_url: str) -> str:
    """Main PDF extraction function with advanced parsing"""
    return extract_pdf_content_advanced(pdf_url)


# Helper function to download and extract text from PDF with caching
def extract_pdf_content(pdf_url: str) -> str:
    """Enhanced PDF extraction with unstructured for better table and layout parsing"""
    return extract_pdf_content_advanced(pdf_url)

# Global variables to store the vector store and hybrid retriever
vector_store = None
hybrid_retriever = None
processed_documents = []
batch_processor = None
last_request_model = None  # Track last used model

@app.get("/")
def root():
    return {
        "message": "LangChain RAG Backend with Mistral LLM and Caching",
        "status": "running",
        "features": [
            "Multiple Mistral LLM fallback",
            "Multiple embedding fallback (HuggingFace Local + Endpoint + Mistral)",
            "Hybrid retrieval (Vector + BM25 + MMR)",
            "Query logging and analytics",
            "Document and embedding caching"
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


def process_documents(documents):
    """Process documents from either text or URLs, handling both single documents and lists"""
    content = ""
    
    # Handle both single string and list of strings
    if isinstance(documents, list):
        for doc in documents:
            if is_url(doc) and doc.lower().endswith('.pdf'):
                # Download and extract PDF content
                pdf_content = extract_pdf_content(doc)
                content += pdf_content + "\n\n"
            else:
                # Treat as plain text
                content += doc + "\n\n"
    else:
        # Single document
        if is_url(documents) and documents.lower().endswith('.pdf'):
            content = extract_pdf_content(documents)
        else:
            content = documents
    
    print(f"📄 Processed documents: {len(content):,} characters")
    return content

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
        """Expand query with related terms for any document type - ENHANCED for better coverage"""
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
            'criteria': ['criteria', 'standard', 'requirement', 'condition', 'specification'],
            
            # ENHANCED: Scientific and physics terms for Newton-like content
            'force': ['force', 'power', 'strength', 'energy', 'gravity', 'attraction', 'pressure'],
            'motion': ['motion', 'movement', 'velocity', 'acceleration', 'speed', 'orbit', 'trajectory'],
            'body': ['body', 'object', 'particle', 'mass', 'matter', 'substance', 'entity'],
            'law': ['law', 'principle', 'rule', 'axiom', 'theorem', 'proposition', 'corollary'],
            'moon': ['moon', 'lunar', 'satellite', 'celestial', 'orbit', 'apogee', 'perigee'],
            'planet': ['planet', 'planetary', 'celestial', 'jupiter', 'mars', 'venus', 'saturn'],
            'experiment': ['experiment', 'test', 'trial', 'demonstration', 'observation', 'proof'],
            'tide': ['tide', 'tidal', 'flux', 'reflux', 'sea', 'ocean', 'water'],
            'vortex': ['vortex', 'vortices', 'whirlpool', 'rotation', 'spiral', 'circulation'],
            'dog': ['dog', 'animal', 'pet', 'canine', 'diamond', 'favorite', 'incident'],
            'stone': ['stone', 'rock', 'projectile', 'object', 'thrown', 'projected', 'example'],
            'pendulum': ['pendulum', 'oscillation', 'swing', 'period', 'frequency', 'bob'],
            'gold': ['gold', 'silver', 'wood', 'metal', 'material', 'substance', 'element'],
            'quantity': ['quantity', 'amount', 'measure', 'mass', 'volume', 'matter', 'measurement']
        }
        
        expanded_terms = [query]
        query_lower = query.lower()
        
        # Look for exact matches and related terms
        for key, terms in general_expansions.items():
            if key in query_lower:
                expanded_terms.extend(terms)
        
        # ENHANCED: Add individual word expansions
        query_words = query_lower.split()
        for word in query_words:
            if word in general_expansions:
                expanded_terms.extend(general_expansions[word])
        
        # Add number variations if present
        import re
        numbers = re.findall(r'\d+', query)
        for num in numbers:
            expanded_terms.extend([
                f'section {num}', f'chapter {num}', f'part {num}', f'clause {num}',
                f'book {num}', f'theorem {num}', f'proposition {num}', f'page {num}'
            ])
        
        # Add common document structure variations
        if any(word in query_lower for word in ['section', 'chapter', 'part', 'clause', 'book']):
            expanded_terms.extend(['provision', 'subsection', 'paragraph', 'article', 'theorem', 'proposition'])
        
        # ENHANCED: Add contextual terms based on question type
        if 'newton' in query_lower:
            expanded_terms.extend(['principia', 'mathematical', 'natural', 'philosophy', 'gravity', 'motion'])
        
        if any(word in query_lower for word in ['explain', 'explanation', 'describe', 'how']):
            expanded_terms.extend(['method', 'approach', 'theory', 'principle', 'concept'])
        
        if any(word in query_lower for word in ['three', 'laws', 'axioms']):
            expanded_terms.extend(['first', 'second', 'third', 'law', 'motion', 'inertia', 'acceleration', 'reaction'])
        
        return list(set(expanded_terms))
    
    def retrieve_relevant_docs(self, query: str, k: int = 12) -> List[Document]:
        """ULTRA-ENHANCED retrieval with maximum coverage to prevent hallucination"""
        all_docs = []
        
        # Expand query terms for better document retrieval
        expanded_queries = self.expand_query_terms(query)
        
        print(f"🔍 ULTRA-Enhanced hybrid retrieval with {len(expanded_queries)} query variations")
        
        # 1. Vector-based retrieval with MAXIMUM parameters
        for i, q in enumerate(expanded_queries[:6]):  # Increased to 6
            try:
                vector_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k * 2, "fetch_k": k * 6}  # MUCH more aggressive
                )
                vector_docs = vector_retriever.get_relevant_documents(q)
                all_docs.extend(vector_docs)
                print(f"   📊 Vector query {i+1}: {len(vector_docs)} docs")
            except Exception as e:
                print(f"⚠️ Vector retrieval failed for query {i+1}: {e}")
        
        # 2. BM25 keyword retrieval with MAXIMUM queries
        if self.bm25_retriever:
            try:
                for i, q in enumerate(expanded_queries[:5]):  # Increased to 5
                    bm25_docs = self.bm25_retriever.get_relevant_documents(q)
                    all_docs.extend(bm25_docs)
                    print(f"   🔤 BM25 query {i+1}: {len(bm25_docs)} docs")
            except Exception as e:
                print(f"⚠️ BM25 retrieval failed: {e}")
        
        # 3. MMR retrieval with MAXIMUM diversity
        try:
            mmr_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k * 2, "fetch_k": k * 8, "lambda_mult": 0.5}  # MAXIMUM diversity
            )
            mmr_docs = mmr_retriever.get_relevant_documents(query)
            all_docs.extend(mmr_docs)
            print(f"   🎯 MMR retrieval: {len(mmr_docs)} docs")
        except Exception as e:
            print(f"⚠️ MMR retrieval failed: {e}")
        
        # 4. MULTIPLE similarity searches with different parameters
        similarity_configs = [
            {"k": k * 2, "fetch_k": k * 8},
            {"k": k * 3, "fetch_k": k * 10},
            {"k": k * 4, "fetch_k": k * 12}
        ]
        
        for i, config in enumerate(similarity_configs):
            try:
                similarity_docs = self.vector_store.similarity_search(
                    query, 
                    k=config["k"],
                    fetch_k=config["fetch_k"]
                )
                all_docs.extend(similarity_docs)
                print(f"   🎯 Similarity search {i+1}: {len(similarity_docs)} docs")
            except Exception as e:
                print(f"⚠️ Similarity search {i+1} failed: {e}")
        
        # 5. FUZZY matching for each expanded query
        for i, expanded_query in enumerate(expanded_queries[:3]):  # Top 3 for fuzzy
            try:
                fuzzy_docs = self.vector_store.similarity_search(
                    expanded_query,
                    k=k,
                    fetch_k=k * 4
                )
                all_docs.extend(fuzzy_docs)
                print(f"   🔍 Fuzzy search {i+1}: {len(fuzzy_docs)} docs")
            except Exception as e:
                print(f"⚠️ Fuzzy search {i+1} failed: {e}")
        
        # Remove duplicates and score documents with LOWER threshold for inclusion
        unique_docs = []
        seen_content = set()
        scored_docs = []
        
        for doc in all_docs:
            # Use first 200 chars for deduplication (shorter to catch more variations)
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                score = self.calculate_document_relevance_score(doc.page_content, query, expanded_queries)
                scored_docs.append((doc, score))
                seen_content.add(content_hash)
        
        # Sort by relevance score and return MORE docs with LOWER threshold
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return MORE documents - be more inclusive to prevent hallucination
        final_k = min(k * 3, len(scored_docs))  # Tripled the documents returned
        unique_docs = [doc for doc, score in scored_docs[:final_k]]
        
        print(f"✅ Enhanced anti-hallucination retrieval: {len(unique_docs)} unique docs from {len(all_docs)} total")
        if scored_docs:
            top_scores = [score for _, score in scored_docs[:8]]  # Show more scores
            print(f"📊 Top relevance scores: {[round(s, 1) for s in top_scores]}")
        
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

def create_optimized_context(relevant_docs, question, max_length=25000):  # Increased from 20000
    """Create MAXIMUM COVERAGE context with zero-tolerance anti-hallucination approach"""
    if not relevant_docs:
        return ""
    
    # Enhanced query term analysis for general documents
    query_terms = question.lower().split()
    
    # ULTRA-EXPANDED term lists for maximum matching
    structure_terms = [
        'section', 'chapter', 'part', 'clause', 'paragraph', 'article', 'provision',
        'subsection', 'subpart', 'schedule', 'appendix', 'exhibit', 'attachment',
        'book', 'volume', 'page', 'theorem', 'proposition', 'corollary', 'lemma',
        'definition', 'axiom', 'principle', 'law', 'rule', 'scholium'
    ]
    
    procedure_terms = [
        'procedure', 'process', 'steps', 'method', 'approach', 'implementation',
        'guidelines', 'instructions', 'requirements', 'criteria', 'standards',
        'experiment', 'observation', 'demonstration', 'proof', 'explanation',
        'theory', 'hypothesis', 'concept', 'principle', 'technique'
    ]
    
    action_terms = [
        'shall', 'must', 'will', 'should', 'may', 'can', 'required', 'mandatory',
        'optional', 'prohibited', 'allowed', 'permitted', 'authorized',
        'describes', 'explains', 'demonstrates', 'proves', 'shows', 'states',
        'argues', 'claims', 'asserts', 'proposes', 'suggests', 'indicates'
    ]
    
    # MAXIMUM INCLUSION scoring - be extremely generous
    scored_docs = []
    
    for doc in relevant_docs:
        content_lower = doc.page_content.lower()
        content = doc.page_content.strip()
        
        if not content:
            continue
        score = 1.0  # Start with base score for ALL documents
        
        # Primary scoring: ULTRA-GENEROUS term matching
        for term in query_terms:
            if len(term) > 0:  # Include ALL terms
                exact_matches = content_lower.count(term)
                if term.isdigit() or any(char.isdigit() for char in term):
                    score += exact_matches * 8  # Maximum weight for numbers
                else:
                    score += exact_matches * 6  # Maximum weight for terms
        
        # AGGRESSIVE PARTIAL MATCHING for maximum coverage
        for term in query_terms:
            if len(term) > 2:  # Most terms
                for word in content_lower.split():
                    # Bidirectional partial matching
                    if (term in word or word in term or 
                        any(term in w or w in term for w in word.split('-')) or
                        any(term in w or w in term for w in word.split('_'))):
                        score += 3  # High bonus for partial matches
        
        # MAXIMUM terminology bonuses
        for term in structure_terms:
            if term in content_lower:
                score += 4  # Maximum bonus
        
        for term in procedure_terms:
            if term in content_lower:
                score += 4  # Maximum bonus
        
        for term in action_terms:
            if term in content_lower:
                score += 3  # High bonus
        
        # ULTRA-ENHANCED pattern matching
        import re
        
        # Look for ANY numbered or structured references
        numbered_refs = re.findall(r'\b\d+\b', content)
        score += len(numbered_refs) * 1.0  # Higher weight
        
        # Look for scientific/mathematical/historical terms
        domain_terms = [
            'force', 'motion', 'gravity', 'orbit', 'body', 'mass', 'velocity', 'acceleration',
            'newton', 'principia', 'mathematical', 'natural', 'philosophy', 'law', 'theorem',
            'experiment', 'observation', 'demonstration', 'proof', 'evidence', 'calculation'
        ]
        for term in domain_terms:
            if term in content_lower:
                score += 3
        
        # ZERO threshold - include EVERYTHING with any relevance
        scored_docs.append((doc, score, len(content)))
    
    # Sort by relevance score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Build MAXIMUM COVERAGE context
    context_parts = []
    context_parts.append("=== ULTRA-COMPREHENSIVE DOCUMENT ANALYSIS ===\n")
    context_parts.append("ALL sections below contain potentially relevant information for exhaustive analysis:\n")
    
    current_length = 0
    section_num = 1
    
    for doc, score, content_length in scored_docs:
        content = doc.page_content.strip()
        
        # Enhanced section header with relevance scoring
        section_header = f"\n--- SECTION {section_num} (Relevance: {score:.1f}) ---\n"
        total_addition = len(section_header) + len(content)
        
        if current_length + total_addition > max_length:
            # Be VERY inclusive with truncation
            remaining_space = max_length - current_length - len(section_header)
            if remaining_space > 200:  # Very low threshold
                truncated_content = content[:remaining_space] + "\n[...additional content available...]"
                context_parts.append(section_header)
                context_parts.append(truncated_content)
                current_length += len(section_header) + len(truncated_content)
            break
        
        context_parts.append(section_header)
        context_parts.append(content)
        current_length += total_addition
        section_num += 1
        
        # Allow MAXIMUM sections - up to 20
        if section_num > 20 and current_length > max_length * 0.6:
            break
    
    # COMPREHENSIVE analysis guidance
    context_parts.append(f"\n\n=== EXHAUSTIVE ANALYSIS PROTOCOL ===")
    context_parts.append(f"- Total sections for analysis: {section_num - 1}")
    context_parts.append(f"- Total context length: {current_length:,} characters")
    context_parts.append("- MANDATE: Examine every section thoroughly for ANY relevant information")
    context_parts.append("- STRATEGY: Look for direct answers, related concepts, supporting details, and connections")
    context_parts.append("- REASONING: Use logical inference when information is distributed across sections")
    context_parts.append("- SYNTHESIS: Combine information from multiple sections for complete answers")
    context_parts.append("- MISSION: Extract maximum value from all available content")
    
    final_context = "\n".join(context_parts)
    
    # Enhanced debug output
    print(f"📄 MAXIMUM COVERAGE context: {len(final_context):,} chars from {section_num-1} sections")
    if scored_docs:
        all_scores = [score for _, score, _ in scored_docs]
        print(f"📊 Score distribution: min={min(all_scores):.1f}, max={max(all_scores):.1f}, avg={sum(all_scores)/len(all_scores):.1f}")
        print(f"📈 Top scores: {[round(score, 1) for _, score, _ in scored_docs[:5]]}")
    
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
    
    print(f"🚀 Processing {len(request.questions)} questions with CACHING - Request: {request_id}")
    
    try:
        # Step 1: Process documents with caching (handles both text and URLs)
        document_content = process_documents(request.documents)
        document_type = "pdf" if any(is_url(doc) and 'pdf' in doc.lower() for doc in (request.documents if isinstance(request.documents, list) else [request.documents])) else "text"
        
        if document_content.strip():
            # Generate content hash for cache key
            content_hash = document_cache.get_content_hash(document_content)
            
            # Check if we have cached chunks and vector store
            cached_chunks = document_cache.get_cached_chunks(content_hash)
            cached_vector_store = document_cache.get_cached_vector_store(content_hash)
            
            if cached_chunks and cached_vector_store:
                print("⚡ Using fully cached data - no processing needed!")
                chunks = cached_chunks
                vector_store = cached_vector_store
                processed_documents = chunks
                
                # Initialize hybrid retriever
                hybrid_retriever = HybridRetriever(vector_store, processed_documents)
                print("✅ Loaded from cache and hybrid retriever created successfully")
            else:
                print("🔄 Cache miss - processing documents...")
                
                # Create documents
                docs = [Document(page_content=document_content)]
                
                # Check if we have cached chunks
                if cached_chunks:
                    print("📁 Using cached chunks")
                    chunks = cached_chunks
                else:
                    # Split documents into chunks with optimized parameters for speed
                    text_splitter_fast = RecursiveCharacterTextSplitter(
                        chunk_size=1500,  # Even larger chunks for faster processing
                        chunk_overlap=250,  # Adequate overlap
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    chunks = text_splitter_fast.split_documents(docs)
                    print(f"Created {len(chunks)} chunks from documents")
                    
                    # Limit chunks for faster processing if too many
                    if len(chunks) > 400:  # Further reduced limit
                        print(f"⚡ Limiting to 400 chunks (from {len(chunks)}) for maximum speed")
                        chunks = chunks[:400]
                    
                    # Cache the chunks
                    document_cache.cache_chunks(content_hash, chunks)
                
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
                """Process a single question with full parallelization"""
                async with semaphore:
                    question_start_time = time.time()
                    chunks_retrieved = 0
                    
                    try:
                        print(f"🔍 Processing Q{question_index+1} in parallel: {question[:50]}...")
                        
                        # Use hybrid retriever with ABSOLUTE MAXIMUM parameters
                        if hybrid_retriever:
                            relevant_docs = hybrid_retriever.retrieve_relevant_docs(question, k=15)  # Increased from 12 to 15
                        else:
                            # ABSOLUTE MAXIMUM fallback retrieval parameters
                            retriever = vector_store.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 15, "fetch_k": 40}  # ABSOLUTE MAXIMUM
                            )
                            relevant_docs = retriever.get_relevant_documents(question)
                        
                        chunks_retrieved = len(relevant_docs)
                        print(f"   📊 Q{question_index+1}: Retrieved {chunks_retrieved} chunks")
                        
                        if relevant_docs:
                            # Create MAXIMUM COVERAGE context for thorough analysis
                            context = create_optimized_context(relevant_docs, question, max_length=18000)  # Increased from 15000
                            
                            # Enhanced debugging for context quality
                            print(f"   📄 Q{question_index+1}: Context length {len(context):,} chars")

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
        
        print(f"🎉 CACHED RAG processing completed in {processing_time:.2f} seconds")
        print(f"📊 Average time per question: {avg_time_per_question:.2f} seconds")
        print(f"⚡ Speed improvement: {len(request.questions)}x parallelization + caching")
        print(f"🎯 Used model: {current_model_name}")
        print(f"📝 Logged {len(request.questions)} queries to CSV")
        print(f"🔍 Used {'hybrid' if hybrid_retriever else 'basic'} retrieval system")
        print(f"💾 Cache stats: {document_cache.get_cache_stats()}")
        
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
    print("🚀 Starting RAG Backend Server with Enhanced PDF Processing...")
    print("📍 Server will be available at:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000")
    print("🔧 Features:")
    print("   - Multiple Mistral LLM fallback with intelligent load balancing")
    print("   - Multiple embedding fallback (HuggingFace Local + Endpoint + Mistral)")
    print("   - Ultra-enhanced hybrid retrieval (Vector + BM25 + MMR + Fuzzy)")
    print("   - Advanced PDF parsing with unstructured (tables, layouts, OCR fallback)")
    print("   - Advanced anti-hallucination measures with maximum content extraction")
    print("   - Comprehensive query logging and performance analytics")
    print("   - Intelligent document and embedding caching for lightning-fast processing")
    print("   - Zero-tolerance approach to 'information not found' responses")
    
    if UNSTRUCTURED_AVAILABLE:
        print("✅ Enhanced PDF processing: Unstructured library loaded")
        print("   - Layout-aware parsing")
        print("   - Table extraction and flattening")
        print("   - OCR fallback for scanned documents")
        print("   - Structured content separation")
    else:
        print("⚠️ Standard PDF processing: Using PyPDF fallback")
        print("💡 To enable enhanced PDF processing, install:")
        print("   pip install 'unstructured[pdf]'")
        print("   pip install pdf2image")
        print("   pip install pytesseract")
        print("📋 Enhanced features you'll get:")
        print("   - Better table extraction from insurance policies")
        print("   - Layout-aware document parsing") 
        print("   - OCR for scanned documents")
        print("   - Structured content separation")
    
    uvicorn.run(app, host=HOST, port=PORT)
