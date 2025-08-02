from fastapi import FastAPI, HTTPException, Depends, Header
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
from concurrent.futures import ThreadPoolExecutor
import random
import math
import pandas as pd
import csv
from pathlib import Path

class RateLimitManager:
    """Manages API rate limits to prevent 429 errors"""
    
    def __init__(self):
        self.request_counts = {}  # Track requests per provider
        self.last_reset = {}     # Track when counters were reset
        self.rate_limits = {
            'mistral': {
                'requests_per_minute': 100,  # Conservative estimate
                'tokens_per_minute': 400000,  # 80% of 500k limit
                'current_requests': 0,
                'current_tokens': 0
            },
            'groq': {
                'requests_per_minute': 200,
                'tokens_per_minute': 800000,  # Higher Groq limits
                'current_requests': 0,
                'current_tokens': 0
            }
        }
    
    def reset_if_needed(self, provider):
        """Reset counters if a minute has passed"""
        now = time.time()
        if provider not in self.last_reset:
            self.last_reset[provider] = now
            return
        
        if now - self.last_reset[provider] >= 60:  # 1 minute
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

class LLMFallbackManager:
    """Manages multiple LLM models with automatic fallback and rate limit handling"""
    
    def __init__(self):
        self.models = []
        self.current_index = 0
        self.rate_limit_delays = {}  # Track models that hit rate limits
        self.rate_manager = RateLimitManager()  # Add rate limit manager
        self.request_count = 0  # Track total requests for rotation
        self.last_used_models = []  # Track recently used models to avoid repeating
        
        # Add ALL Mistral models as PRIMARY options (comprehensive fallback chain)
        if MISTRAL_API_KEY:
            try:
                from langchain_mistralai import ChatMistralAI
                
                # Mistral models in order of preference (latest and best first)
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
                
                print(f"✅ Mistral LLMs initialized successfully: {len(self.models)} models")
                print("🎯 Primary model: mistral-large-2411 (latest and best)")
            except Exception as e:
                print(f"⚠️ Error initializing Mistral LLMs: {e}")
        
        # Add Groq LLMs as final fallback if API key is available
        if GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                # Llama 3.3 as first Groq fallback
                llama33_llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=GROQ_API_KEY,
                    temperature=0.2,
                    max_tokens=2000
                )
                self.models.append({
                    "name": "llama-3.3-70b-versatile",
                    "provider": "groq",
                    "model": llama33_llm
                })
                
                # Add Llama 3.1 as additional backup
                llama31_llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    groq_api_key=GROQ_API_KEY,
                    temperature=0.2,
                    max_tokens=2000
                )
                self.models.append({
                    "name": "llama-3.1-8b-instant",
                    "provider": "groq",
                    "model": llama31_llm
                })
                
                print(f"✅ Groq LLMs initialized as emergency fallback: added {len(self.models)} models total")
            except Exception as e:
                print(f"⚠️ Error initializing Groq LLM: {e}")
        
        if not self.models:
            print("❌ No LLM models could be initialized. Please check your API keys.")
        else:
            print("🔄 Model priority order (comprehensive Mistral chain):")
            for i, model in enumerate(self.models, 1):
                priority_label = "PRIMARY" if i == 1 else "FALLBACK" if i <= len(self.models) - 2 else "EMERGENCY"
                print(f"   {i}. {model['name']} ({model['provider']}) - {priority_label}")

    def get_current_model_info(self):
        """Get information about the currently active model"""
        if not self.models:
            return {"name": "none", "provider": "none"}
        return {
            "name": self.models[self.current_index]["name"],
            "provider": self.models[self.current_index]["provider"]
        }
    
    def get_next_available_model(self):
        """Get the next available model, skipping rate-limited ones"""
        if not self.models:
            raise ValueError("No LLM models available")
        
        # Check all models starting from current
        start_index = self.current_index
        
        # First pass: try to find a model that's not rate-limited
        for _ in range(len(self.models)):
            model_info = self.models[self.current_index]
            model_key = f"{model_info['provider']}:{model_info['name']}"
            
            # If model is not rate-limited or delay has passed
            if model_key not in self.rate_limit_delays:
                return model_info["model"]
            
            # Check if rate limit delay has expired
            if time.time() > self.rate_limit_delays[model_key]:
                del self.rate_limit_delays[model_key]
                return model_info["model"]
            
            # Try next model
            self.current_index = (self.current_index + 1) % len(self.models)
        
        # Second pass: if all models are rate-limited, use the one with the shortest delay
        min_delay = float('inf')
        best_model = None
        
        for i, model_info in enumerate(self.models):
            model_key = f"{model_info['provider']}:{model_info['name']}"
            if model_key in self.rate_limit_delays:
                remaining_delay = self.rate_limit_delays[model_key] - time.time()
                if remaining_delay < min_delay:
                    min_delay = remaining_delay
                    best_model = model_info["model"]
                    self.current_index = i
        
        # If all models are rate-limited, sleep for the minimum delay
        if min_delay > 0 and min_delay != float('inf'):
            print(f"⏳ All models rate-limited. Waiting for {min_delay:.2f} seconds...")
            time.sleep(min_delay + 0.5)  # Add a small buffer
        
        return best_model
    
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
    
    def rotate_to_next_available_model(self):
        """Intelligently rotate to next available model to distribute load"""
        if not self.models or len(self.models) == 1:
            return
        
        # Find the next non-rate-limited model that wasn't recently used
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

    def select_best_model_for_request(self):
        """Select the best model for a new request to avoid rate limits"""
        if not self.models:
            return None
        
        # Increment request counter for rotation logic
        self.request_count += 1
        
        # For new requests, try to use a different model than previous requests
        if self.request_count > 1:
            # Look for available models, prioritizing those not recently used
            available_models = []
            
            for i, model_info in enumerate(self.models):
                model_key = f"{model_info['provider']}:{model_info['name']}"
                
                # Skip rate-limited models
                if model_key in self.rate_limit_delays:
                    if time.time() <= self.rate_limit_delays[model_key]:
                        continue
                    else:
                        # Rate limit expired, remove from tracking
                        del self.rate_limit_delays[model_key]
                
                available_models.append(i)
            
            if available_models:
                # Prefer models that haven't been used recently
                for model_idx in available_models:
                    if model_idx != self.current_index:
                        self.current_index = model_idx
                        model_name = self.models[model_idx]['name']
                        print(f"🎯 Selected fresh model for new request: {model_name}")
                        return self.models[model_idx]["model"]
                
                # If no different model available, use any available model
                if available_models:
                    self.current_index = available_models[0]
        
        return self.models[self.current_index]["model"]

    def invoke(self, prompt):
        """Invoke the LLM with fallback handling"""
        if not self.models:
            raise ValueError("No LLM models available")
        
        errors = []
        
        # Try each model until one succeeds
        for _ in range(len(self.models) * 2):  # Allow two passes through all models
            try:
                model = self.get_next_available_model()
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
                    self.mark_rate_limited(current_index, delay_seconds=300)  # 5 minutes
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
                    self.mark_rate_limited(self.current_index, delay_seconds=120)  # Longer delay
                    errors.append(f"Rate limit for {model_info['name']}")
                else:
                    errors.append(f"Error with {model_info['name']}: {str(e)}")
                    
                # Move to next model
                self.current_index = (self.current_index + 1) % len(self.models)
        
        # All models failed
        error_msg = "; ".join(errors)
        raise RuntimeError(f"All LLM models failed: {error_msg}")

class EmbeddingsFallbackManager:
    """Manages fallback between different embedding providers with rate limit handling"""
    
    def __init__(self):
        self.embeddings = []
        self.current_index = 0
        self.rate_limit_delays = {}  # Track embeddings that hit rate limits
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        """Initialize available embeddings in order of preference"""
        self.embeddings = []
        
        # Mistral embeddings (primary)
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
                print(f"⚠️ Mistral embeddings initialization failed: {e}")
        
        # HuggingFace local embeddings (fallback)
        try:
            local_emb = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.embeddings.append({
                "name": "all-MiniLM-L6-v2",
                "provider": "huggingface_local",
                "embedding": local_emb
            })
            print("✅ HuggingFace local embeddings initialized")
        except Exception as e:
            print(f"⚠️ HuggingFace local embeddings failed: {e}")
        
        # HuggingFace endpoint embeddings (last resort)
        if HF_TOKEN:
            try:
                endpoint_emb = HuggingFaceEndpointEmbeddings(
                    model="sentence-transformers/all-mpnet-base-v2",
                    huggingfacehub_api_token=HF_TOKEN
                )
                self.embeddings.append({
                    "name": "all-mpnet-base-v2",
                    "provider": "huggingface_endpoint",
                    "embedding": endpoint_emb
                })
                print("✅ HuggingFace Endpoint embeddings initialized")
            except Exception as e:
                print(f"⚠️ HuggingFace Endpoint embeddings failed: {e}")
        
        if not self.embeddings:
            print("❌ No embedding models available!")
        else:
            print(f"✅ Initialized {len(self.embeddings)} embedding models for fallback")
    
    def is_rate_limited(self, embedding_name: str) -> bool:
        """Check if an embedding model is currently rate limited"""
        if embedding_name in self.rate_limit_delays:
            delay_until = self.rate_limit_delays[embedding_name]
            if time.time() < delay_until:
                return True
            else:
                # Rate limit period expired, remove from tracking
                del self.rate_limit_delays[embedding_name]
        return False
    
    def set_rate_limit(self, embedding_name: str, delay_seconds: int = 60):
        """Mark an embedding model as rate limited for specified duration"""
        self.rate_limit_delays[embedding_name] = time.time() + delay_seconds
        print(f"⏰ Embedding {embedding_name} rate limited for {delay_seconds} seconds")
    
    def get_next_available_embedding(self):
        """Get the next available embedding model that isn't rate limited"""
        if not self.embeddings:
            return None
        
        # Try to find a non-rate-limited embedding
        for i in range(len(self.embeddings)):
            embedding_index = (self.current_index + i) % len(self.embeddings)
            embedding_info = self.embeddings[embedding_index]
            
            if not self.is_rate_limited(embedding_info["name"]):
                self.current_index = embedding_index
                return embedding_info
        
        # All embeddings are rate limited, return the one with least remaining delay
        min_delay_embedding = min(self.embeddings, 
                                key=lambda e: self.rate_limit_delays.get(e["name"], 0))
        return min_delay_embedding
    
    def embed_documents(self, texts):
        """Embed documents with automatic fallback"""
        if not self.embeddings:
            raise ValueError("No embedding models available")
        
        for attempt in range(len(self.embeddings)):
            embedding_info = self.get_next_available_embedding()
            
            if not embedding_info:
                raise Exception("No embedding models available")
            
            embedding_name = embedding_info["name"]
            embedding_model = embedding_info["embedding"]
            
            try:
                print(f"🔤 Attempting embedding with: {embedding_name}")
                
                # Check if we need to wait for rate limit
                if self.is_rate_limited(embedding_name):
                    wait_time = self.rate_limit_delays[embedding_name] - time.time()
                    if wait_time > 0 and wait_time < 10:  # Only wait if it's less than 10 seconds
                        print(f"⏳ Waiting {wait_time:.1f}s for embedding rate limit to expire...")
                        time.sleep(wait_time)
                
                # Try to embed
                result = embedding_model.embed_documents(texts)
                print(f"✅ Success with embedding: {embedding_name}")
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle rate limiting
                if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    print(f"🚫 Rate limit hit for embedding {embedding_name}")
                    delay = min(300, 60 * (2 ** attempt))  # Exponential backoff, max 5 minutes
                    self.set_rate_limit(embedding_name, delay)
                    
                    # Move to next embedding
                    self.current_index = (self.current_index + 1) % len(self.embeddings)
                    continue
                
                else:
                    print(f"❌ Error with embedding {embedding_name}: {e}")
                    
                    # For other errors, try next embedding
                    if attempt < len(self.embeddings) - 1:
                        self.current_index = (self.current_index + 1) % len(self.embeddings)
                        continue
        
        # All embeddings failed
        raise Exception(f"All embedding models failed after {len(self.embeddings)} attempts")
    
    def embed_query(self, text):
        """Embed a query with automatic fallback"""
        if not self.embeddings:
            raise ValueError("No embedding models available")
        
        for attempt in range(len(self.embeddings)):
            embedding_info = self.get_next_available_embedding()
            
            if not embedding_info:
                raise Exception("No embedding models available")
            
            embedding_name = embedding_info["name"]
            embedding_model = embedding_info["embedding"]
            
            try:
                # Check if we need to wait for rate limit
                if self.is_rate_limited(embedding_name):
                    wait_time = self.rate_limit_delays[embedding_name] - time.time()
                    if wait_time > 0 and wait_time < 10:
                        time.sleep(wait_time)
                
                # Try to embed
                result = embedding_model.embed_query(text)
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle rate limiting
                if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                    delay = min(300, 60 * (2 ** attempt))
                    self.set_rate_limit(embedding_name, delay)
                    self.current_index = (self.current_index + 1) % len(self.embeddings)
                    continue
                
                else:
                    if attempt < len(self.embeddings) - 1:
                        self.current_index = (self.current_index + 1) % len(self.embeddings)
                        continue
        
        raise Exception(f"All embedding models failed after {len(self.embeddings)} attempts")
    
    def get_current_embedding_info(self) -> Dict:
        """Get information about currently selected embedding"""
        if not self.embeddings:
            return {"name": "none", "provider": "none", "available": False}
        
        embedding_info = self.embeddings[self.current_index]
        return {
            "name": embedding_info["name"],
            "provider": embedding_info["provider"], 
            "available": not self.is_rate_limited(embedding_info["name"]),
            "total_embeddings": len(self.embeddings),
            "rate_limited_embeddings": len(self.rate_limit_delays)
        }

# Load environment variables from .env file
load_dotenv()

# Only suppress tokenizer parallelism warnings (still useful for HuggingFace)
os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'false')

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing dependencies:")
    print("pip install langchain langchain-community langchain-groq langchain-huggingface faiss-cpu pypdf requests rank-bm25")
    exit(1)

# Enhanced request and response models for insurance claim processing
class DebugRequest(BaseModel):
    question: str

class ClaimRequest(BaseModel):
    documents: Union[List[str], str]
    claim_details: Dict[str, Any] = Field(default_factory=dict)
    questions: List[str]

# Simple response model to match required format
class AnswerResponse(BaseModel):
    answers: List[str]

class CoordinationOfBenefits(BaseModel):
    has_other_insurance: bool = False
    primary_insurance: Optional[str] = None
    secondary_insurance: Optional[str] = None
    primary_payment: Optional[float] = None
    remaining_amount: Optional[float] = None

class ClaimDecision(BaseModel):
    question: str
    decision: str  # "APPROVED", "DENIED", "PENDING_REVIEW"
    confidence_score: float = Field(ge=0.0, le=1.0)
    payout_amount: Optional[float] = None
    reasoning: str
    policy_sections_referenced: List[str] = Field(default_factory=list)
    exclusions_applied: List[str] = Field(default_factory=list)
    coordination_of_benefits: Optional[CoordinationOfBenefits] = None
    processing_notes: List[str] = Field(default_factory=list)

class ProcessingMetadata(BaseModel):
    request_id: str
    processing_time: float
    chunks_analyzed: int
    model_used: str
    timestamp: str

class EnhancedAnswerResponse(BaseModel):
    decisions: List[ClaimDecision]
    processing_metadata: ProcessingMetadata
    audit_trail: List[str] = Field(default_factory=list)

# Initialize FastAPI application with enhanced documentation
app = FastAPI(
    title="HackRx 6.0 Universal Document Analysis RAG Backend", 
    version="2.0.0",
    description="Advanced RAG Backend with Insurance Claim Decision Engine and Structured Analysis"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

security = HTTPBearer()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HOST = os.getenv("HOST", "0.0.0")
PORT = int(os.getenv("PORT", 5000))

# Enhanced Universal Document Analysis Prompt Template
UNIVERSAL_ANSWER_PROMPT = """
You are an expert document analyst capable of analyzing any type of document. Your task is to provide precise, actionable answers based on the document context provided.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANALYSIS INSTRUCTIONS:
1. **DOCUMENT TYPE DETECTION**: First identify the type of document (legal, insurance, technical, academic, etc.)
2. **DIRECT ANSWER FIRST**: Start with a clear YES/NO or definitive answer when possible
3. **DOCUMENT REFERENCES**: Always cite specific section numbers, page numbers, or exact quotes
4. **COMPREHENSIVE SEARCH**: Look for related terms, synonyms, and indirect references
5. **PRACTICAL GUIDANCE**: Explain what the reader should do next (if applicable)

SPECIALIZED RESPONSE FORMATS:

**FOR LEGAL DOCUMENTS** (contracts, agreements, laws, regulations):
- **Legal Status**: [Clearly state legal position/rights/obligations]
- **Legal Basis**: [Cite specific clauses, sections, or legal provisions]
- **Legal Implications**: [Explain consequences, requirements, or restrictions]
- **Recommended Action**: [Legal next steps or compliance requirements]
- **Important Notes**: [Disclaimers, limitations, or additional considerations]

**FOR INSURANCE DOCUMENTS** (policies, claims, coverage):
- **Coverage Status**: [Covered/Not Covered/Conditional]
- **Policy Reference**: [Specific section and exact quote]
- **Coverage Details**: [Limits, deductibles, conditions]
- **Claim Process**: [Required steps or procedures]
- **Important Conditions**: [Waiting periods, exclusions, requirements]

**FOR TECHNICAL DOCUMENTS** (manuals, specifications, procedures):
- **Technical Answer**: [Direct technical response]
- **Reference**: [Section, page, or specification number]
- **Technical Details**: [Specifications, requirements, or procedures]
- **Implementation**: [How to apply or use the information]
- **Safety/Compliance**: [Important warnings or standards]

**FOR GENERAL DOCUMENTS** (reports, articles, correspondence):
- **Direct Answer**: [Clear response to the question]
- **Source Reference**: [Page, section, or paragraph reference]
- **Supporting Details**: [Additional relevant information]
- **Context**: [Background or related information]
- **Next Steps**: [Recommended actions or follow-up]

CRITICAL RULES:
- **DOCUMENT-SPECIFIC LANGUAGE**: Use terminology appropriate to the document type
- **ACCURATE CITATIONS**: Always provide exact references to source material
- **CONTEXT AWARENESS**: Tailor response style to document formality and purpose
- **COMPREHENSIVE COVERAGE**: Don't miss related information in other sections
- **PRACTICAL VALUE**: Focus on actionable information relevant to the question

LEGAL DOCUMENT DISCLAIMER (use only for legal documents):
*Note: This analysis is based on the document provided and should not be considered legal advice. Consult with a qualified attorney for legal guidance specific to your situation.*

ANSWER:"""

# Create the enhanced universal prompt template
ANSWER_PROMPT = PromptTemplate(
    template=UNIVERSAL_ANSWER_PROMPT,
    input_variables=["context", "question"]
)

# Enhanced Insurance-Specific Prompt Template (keep for structured analysis)
INSURANCE_CLAIM_PROMPT = """
You are an expert insurance claim processor with deep knowledge of policy terms, coverage rules, and claim evaluation. You must analyze claims systematically and provide structured decisions.

ANALYSIS FRAMEWORK:
1. **Eligibility Assessment**: Determine if the claim is covered under the policy
2. **Coverage Limits**: Identify applicable limits, deductibles, and caps
3. **Coordination of Benefits**: Check for multiple insurance policies and calculate remaining amounts
4. **Exclusion Review**: Identify any policy exclusions that apply
5. **Decision Logic**: Apply business rules to determine approval/denial
6. **Payout Calculation**: Calculate exact amounts considering all factors

RESPONSE FORMAT (Must be valid JSON):
{{
    "decision": "[APPROVED/DENIED/PENDING_REVIEW]",
    "confidence_score": [0.0-1.0],
    "payout_amount": [amount or null],
    "reasoning": "Detailed explanation with specific policy references",
    "policy_sections_referenced": ["section1", "section2"],
    "exclusions_applied": ["exclusion1", "exclusion2"],
    "coordination_of_benefits": {{
        "has_other_insurance": [true/false],
        "primary_insurance": "name or null",
        "secondary_insurance": "name or null", 
        "primary_payment": [amount or null],
        "remaining_amount": [amount or null]
    }},
    "processing_notes": ["note1", "note2"]
}}

IMPORTANT RULES:
- Base decisions ONLY on information in the policy context
- For coordination of benefits, calculate remaining amounts after primary insurance
- Include confidence scores based on clarity of policy language
- Reference specific policy sections in your reasoning
- If information is unclear, use "PENDING_REVIEW" decision

Policy Context:
{context}

Claim Question: {question}

Insurance Analysis (JSON format only):
"""

# Create the enhanced prompt template
ENHANCED_PROMPT = PromptTemplate(
    template=INSURANCE_CLAIM_PROMPT,
    input_variables=["context", "question"]
)

class InsuranceDecisionEngine:
    """Core decision engine for insurance claim processing"""
    
    def __init__(self):
        self.decision_rules = {
            'min_confidence_for_approval': 0.7,
            'max_payout_without_review': 10000,
            'coordination_keywords': [
                'coordination of benefits', 'other insurance', 'secondary claim',
                'primary insurance', 'remaining amount', 'balance claim'
            ],
            'exclusion_keywords': [
                'excluded', 'not covered', 'limitation', 'restriction'
            ]
        }
    
    def extract_financial_amounts(self, text: str) -> List[float]:
        """Extract dollar amounts from text"""
        amounts = re.findall(r'\$?[\d,]+\.?\d*', text)
        return [float(amt.replace('$', '').replace(',', '')) for amt in amounts if amt]
    
    def detect_coordination_of_benefits(self, context: str, question: str) -> bool:
        """Detect if coordination of benefits applies"""
        combined_text = (context + " " + question).lower()
        return any(keyword in combined_text for keyword in self.decision_rules['coordination_keywords'])
    
    def calculate_confidence_score(self, context: str, decision_factors: Dict) -> float:
        """Calculate confidence score based on various factors"""
        score = 0.5  # Base score
        
        # Boost confidence if specific policy sections are mentioned
        if decision_factors.get('policy_sections_referenced'):
            score += 0.2
        
        # Reduce confidence if coordination of benefits is involved
        if decision_factors.get('has_coordination'):
            score -= 0.1
        
        # Boost confidence if clear dollar amounts are present
        if decision_factors.get('has_amounts'):
            score += 0.1
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))

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
            doc_texts = [doc.page_content for doc in self.documents]
            self.bm25_retriever = BM25Retriever.from_texts(doc_texts)
            self.bm25_retriever.k = 12  # Increased for better coverage
            print("✅ BM25 retriever initialized")
        except Exception as e:
            print(f"⚠️ BM25 retriever failed, using vector-only: {e}")
            self.bm25_retriever = None
    
    def expand_query_terms(self, query: str) -> List[str]:
        """Expand query with related terms for better retrieval"""
        query_expansions = {
            'grace period': ['grace period', 'premium payment period', 'payment deadline', 'late payment', 'premium due', 'payment extension', 'premium lapse', 'payment terms', 'waiting period'],
            'premium': ['premium', 'payment', 'contribution', 'installment', 'fee', 'cost'],
            'coverage': ['coverage', 'benefit', 'protection', 'insurance', 'claim', 'payout', 'indemnify'],
            'exclusion': ['exclusion', 'not covered', 'limitation', 'restriction', 'exception', 'excluded'],
            'waiting period': ['waiting period', 'pre-existing', 'moratorium', 'cooling period', 'grace period'],
            'deductible': ['deductible', 'excess', 'co-payment', 'copay', 'self-pay'],
            'renewal': ['renewal', 'continuation', 'extension', 'reinstatement'],
            'claim': ['claim', 'reimbursement', 'settlement', 'payout', 'benefit payment', 'indemnify'],
            'newborn': ['newborn', 'infant', 'baby', 'child', 'dependent', 'new addition', 'birth'],
            'ambulance': ['ambulance', 'emergency transport', 'medical transport', 'air ambulance', 'evacuation'],
            'ayush': ['ayush', 'ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy', 'alternative medicine'],
            'organ': ['organ', 'transplant', 'donation', 'donor', 'kidney', 'liver', 'heart'],
            'icu': ['icu', 'intensive care', 'critical care', 'ccr', 'intensive care unit'],
            'medically necessary': ['medically necessary', 'medical necessity', 'clinically necessary', 'evidence-based', 'standard treatment'],
            'health checkup': ['health checkup', 'medical checkup', 'preventive care', 'wellness', 'screening'],
            'critical illness': ['critical illness', 'stroke', 'heart attack', 'cancer', 'serious illness', 'life-threatening'],
            'stay active': ['stay active', 'wellness', 'fitness', 'steps', 'physical activity', 'health incentive', 'discount']
        }
        
        expanded_terms = [query]
        query_lower = query.lower()
        
        for key, terms in query_expansions.items():
            if key in query_lower:
                expanded_terms.extend(terms)
        
        # Add question-specific expansions
        if 'donor' in query_lower or 'donation' in query_lower:
            expanded_terms.extend(['transplant', 'organ', 'recipient'])
        if 'outpatient' in query_lower:
            expanded_terms.extend(['day care', 'inpatient', 'hospitalization'])
        if 'newborn' in query_lower:
            expanded_terms.extend(['waiting period', 'coverage start', 'enrollment'])
        
        return list(set(expanded_terms))
    
    def retrieve_relevant_docs(self, query: str, k: int = 10) -> List[Document]:
        """Enhanced retrieval with better coverage"""
        all_docs = []
        
        # Expand query terms for better retrieval
        expanded_queries = self.expand_query_terms(query)
        
        # Vector-based retrieval with multiple query variations
        for q in expanded_queries[:4]:  # Use top 4 expanded queries
            vector_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k//3, "fetch_k": k * 2}
            )
            vector_docs = vector_retriever.get_relevant_documents(q)
            all_docs.extend(vector_docs)
        
        # BM25 keyword retrieval with expanded terms
        if self.bm25_retriever:
            try:
                for q in expanded_queries[:3]:  # Use top 3 for BM25
                    bm25_docs = self.bm25_retriever.get_relevant_documents(q)
                    all_docs.extend(bm25_docs)
            except Exception as e:
                print(f"⚠️ BM25 retrieval failed: {e}")
        
        # MMR retrieval for diversity
        try:
            mmr_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k//3, "fetch_k": k * 3, "lambda_mult": 0.7}
            )
            mmr_docs = mmr_retriever.get_relevant_documents(query)
            all_docs.extend(mmr_docs)
        except Exception as e:
            print(f"⚠️ MMR retrieval failed: {e}")
        
        # Remove duplicates and return top k with enhanced scoring
        unique_docs = []
        seen_content = set()
        scored_docs = []
        
        for doc in all_docs:
            content_hash = hash(doc.page_content[:200])  # Use first 200 chars for dedup
            if content_hash not in seen_content:
                score = self.calculate_relevance_score(doc.page_content, query, expanded_queries)
                scored_docs.append((doc, score))
                seen_content.add(content_hash)
        
        # Sort by relevance score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        unique_docs = [doc for doc, score in scored_docs[:k * 2]]  # Get more docs for better context
        
        return unique_docs
    
    def calculate_relevance_score(self, content: str, original_query: str, expanded_queries: List[str]) -> float:
        """Enhanced relevance scoring"""
        content_lower = content.lower()
        score = 0.0
        
        # Score based on original query terms (highest weight)
        original_terms = original_query.lower().split()
        for term in original_terms:
            if len(term) > 2:  # Ignore very short terms
                if term in content_lower:
                    score += 3.0
        
        # Score based on expanded query terms
        for query in expanded_queries[1:]:  # Skip first as it's the original
            query_terms = query.lower().split()
            for term in query_terms:
                if len(term) > 2 and term in content_lower:
                    score += 1.5
        
        # Bonus for policy-specific terms
        policy_terms = ['section', 'clause', 'condition', 'provision', 'benefit', 'coverage', 'exclusion', 'indemnify']
        for term in policy_terms:
            if term in content_lower:
                score += 1.0
        
        # Bonus for numbers (often important in insurance)
        import re
        numbers = re.findall(r'\d+', content)
        if numbers:
            score += len(numbers) * 0.2
        
        return score

# Initialize components with enhanced error handling
try:
    from langchain_mistralai import MistralAIEmbeddings
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY
    )
    print("✅ Mistral embeddings initialized successfully")
except Exception as e:
    print(f"⚠️ Mistral embeddings initialization failed: {e}")
    try:
        embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        print("✅ HuggingFace local embeddings initialized successfully")
    except Exception as e2:
        print(f"⚠️ HuggingFace local embeddings failed: {e2}")
        try:
            embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            huggingfacehub_api_token=HF_TOKEN
            )
            print("✅ HuggingFace Endpoint Embeddings initialized successfully")
        except Exception as e3:
            print(f"❌ Error initializing all embedding options: {e3}")
            embeddings = None

# Initialize LLM with fallback management
try:
    llm_manager = LLMFallbackManager()
    if llm_manager.models:
        llm = llm_manager  # Use the manager as the LLM interface
        print("✅ LLM Fallback Manager initialized successfully")
    else:
        llm = None
        print("❌ No LLM models available")
except Exception as e:
    print(f"❌ Error initializing LLM Fallback Manager: {e}")
    llm = None

# Enhanced text splitter for insurance documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Reduced from 1000 for more granular chunks
    chunk_overlap=300,  # Increased from 200 for better context preservation
    separators=[
        "\n\n### ",  # Policy sections
        "\n\nSection ",  # Section breaks
        "\n\nClause ",   # Clause breaks
        "\n\nArticle ",  # Article breaks
        "\n\nCondition ",  # Condition breaks
        "\n\nDefinition ",  # Definition breaks
        "\n\nBenefit ",  # Benefit breaks
        "\n\n",          # Paragraphs
        "\n",            # Lines
        ". ",            # Sentences
        " ",             # Words
    ],
    length_function=len,
    keep_separator=True
)

class BatchProcessor:
    """Handles batch processing of questions with intelligent rate limiting"""
    
    def __init__(self, llm_manager, prompt_template, max_batch_size=3):  # Reduced batch size
        self.llm_manager = llm_manager
        self.prompt_template = prompt_template
        self.max_batch_size = max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=3)  # Reduced workers
        self.semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
    
    def clean_response(self, response_text):
        """Clean up LLM response text"""
        # Handle AIMessage object
        if hasattr(response_text, 'content'):
            response_text = response_text.content
        elif not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Remove common prefixes that LLMs sometimes add
        cleaned = re.sub(r'^(Answer:|ANSWER:|Response:|Here\'s the answer:)\s*', '', response_text.strip())
        return cleaned.strip()
    
    async def process_question(self, question, context):
        """Process a single question with the given context using fallback manager"""
        try:
            formatted_prompt = self.prompt_template.format(context=context, question=question)
            
            # Use fallback manager instead of direct LLM invocation
            if hasattr(self.llm_manager, 'invoke_with_fallback'):
                response = await self.llm_manager.invoke_with_fallback(formatted_prompt)
            else:
                # Fallback to direct invocation for backward compatibility
                response = await asyncio.to_thread(self.llm_manager.invoke, formatted_prompt)
            
            # Extract content from AIMessage object
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            return self.clean_response(response_text)
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            return f"Error answering question: {str(e)}"
    
    async def process_questions_parallel(self, questions, context):
        """Process all questions in parallel for maximum speed"""
        print(f"🚀 Processing {len(questions)} questions in parallel")
        
        # Create tasks for all questions simultaneously
        tasks = [self.process_question(q, context) for q in questions]
        
        # Process all questions concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        print(f"⚡ Parallel processing completed in {processing_time:.2f} seconds")
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(f"Error processing question {i+1}: {str(result)}")
            else:
                final_results.append(result)
        
        return final_results

class QueryLogger:
    """Logs queries, documents, and answers to CSV for analysis"""
    
    def __init__(self, log_file="query_logs.csv"):
        self.log_file = Path(log_file)
        self.ensure_log_file_exists()
    
    def ensure_log_file_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.log_file.exists():
            headers = [
                'timestamp',
                'request_id', 
                'question',
                'document_links',
                'document_type',
                'answer',
                'model_used',
                'processing_time_seconds',
                'chunks_retrieved',
                'success',
                'error_message'
            ]
            
            with open(self.log_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
            
            print(f"✅ Created query log file: {self.log_file}")
    
    def log_query(self, request_id, question, document_links, document_type, answer, 
                  model_used, processing_time, chunks_retrieved=0, success=True, error_message=""):
        """Log a single query and its response"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Clean and prepare data
            question_clean = question.replace('\n', ' ').replace('\r', ' ')[:500]  # Limit length
            answer_clean = answer.replace('\n', ' ').replace('\r', ' ')[:1000] if answer else ""
            links_str = "|".join(document_links) if isinstance(document_links, list) else str(document_links)
            
            row = [
                timestamp,
                request_id,
                question_clean,
                links_str,
                document_type,
                answer_clean,
                model_used,
                round(processing_time, 2),
                chunks_retrieved,
                success,
                error_message
            ]
            
            with open(self.log_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)
                
        except Exception as e:
            print(f"⚠️ Error logging query: {e}")
    
    def get_stats(self):
        """Get statistics from logged queries"""
        try:
            if not self.log_file.exists():
                return {"error": "No log file found"}
            
            df = pd.read_csv(self.log_file)
            
            stats = {
                "total_queries": len(df),
                "successful_queries": len(df[df['success'] == True]),
                "failed_queries": len(df[df['success'] == False]),
                "avg_processing_time": df['processing_time_seconds'].mean(),
                "most_used_model": df['model_used'].mode().iloc[0] if not df.empty else "N/A",
                "document_types": df['document_type'].value_counts().to_dict(),
                "success_rate": (len(df[df['success'] == True]) / len(df) * 100) if len(df) > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            return {"error": f"Error getting stats: {e}"}

# Initialize query logger
query_logger = QueryLogger("d:\\GenerativeAI\\RAG BAJAJ\\query_logs.csv")

def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def create_optimized_context(relevant_docs, question, max_length=8000):
    """Create optimized context with length limits for faster processing"""
    if not relevant_docs:
        return ""
    
    # Sort docs by relevance score if available
    query_terms = question.lower().split()
    scored_docs = []
    
    for doc in relevant_docs:
        content_lower = doc.page_content.lower()
        score = sum(1 for term in query_terms if term in content_lower)
        scored_docs.append((doc, score))
    
    # Sort by relevance and take top docs
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Build context efficiently
    context_parts = []
    current_length = 0
    
    for doc, score in scored_docs:
        content = doc.page_content.strip()
        if not content:
            continue
        
        # Check if adding this content would exceed limit
        if current_length + len(content) > max_length:
            # Truncate the content to fit
            remaining_space = max_length - current_length
            if remaining_space > 100:  # Only add if meaningful space left
                content = content[:remaining_space] + "..."
                context_parts.append(content)
            break
        
        context_parts.append(content)
        current_length += len(content)
    
    return "\n\n".join(context_parts)

def create_enhanced_context(relevant_docs, question, max_length=10000):
    """Create enhanced context with better organization and document type detection"""
    if not relevant_docs:
        return ""
    
    # Detect document type based on content
    all_content = " ".join([doc.page_content for doc in relevant_docs[:5]]).lower()
    
    doc_type = "general"
    if any(term in all_content for term in ['policy', 'coverage', 'premium', 'claim', 'deductible', 'beneficiary']):
        doc_type = "insurance"
    elif any(term in all_content for term in ['contract', 'agreement', 'clause', 'whereas', 'party', 'obligations', 'terms and conditions']):
        doc_type = "legal"
    elif any(term in all_content for term in ['specification', 'procedure', 'manual', 'technical', 'operation', 'maintenance']):
        doc_type = "technical"
    elif any(term in all_content for term in ['constitution', 'article', 'amendment', 'section', 'parliament', 'legislature']):
        doc_type = "legal"
    
    # Sort docs by relevance score
    query_terms = question.lower().split()
    scored_docs = []
    
    for doc in relevant_docs:
        content_lower = doc.page_content.lower()
        # Enhanced scoring considering term frequency and proximity
        score = 0
        for term in query_terms:
            if len(term) > 2:
                count = content_lower.count(term)
                score += count * 2
        
        # Bonus for having multiple query terms
        terms_found = sum(1 for term in query_terms if len(term) > 2 and term in content_lower)
        if terms_found > 1:
            score += terms_found * 3
        
        scored_docs.append((doc, score))
    
    # Sort by relevance
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Build context with clear sections
    context_parts = []
    
    # Document type specific header
    if doc_type == "insurance":
        context_parts.append("=== INSURANCE POLICY DOCUMENT ===\n")
    elif doc_type == "legal":
        context_parts.append("=== LEGAL DOCUMENT ===\n")
    elif doc_type == "technical":
        context_parts.append("=== TECHNICAL DOCUMENT ===\n")
    else:
        context_parts.append("=== DOCUMENT SECTIONS ===\n")
    
    current_length = 0
    section_num = 1
    
    for doc, score in scored_docs:
        content = doc.page_content.strip()
        if not content:
            continue
        
        # Check length limit
        if current_length + len(content) > max_length:
            remaining_space = max_length - current_length
            if remaining_space > 200:  # Only add if meaningful space
                content = content[:remaining_space] + "\n[Content truncated...]"
                context_parts.append(f"\n--- SECTION {section_num} (Relevance: {score:.1f}) ---")
                context_parts.append(content)
            break
        
        context_parts.append(f"\n--- SECTION {section_num} (Relevance: {score:.1f}) ---")
        context_parts.append(content)
        current_length += len(content)
        section_num += 1
    
    # Document type specific footer
    context_parts.append(f"\n\n=== ANALYSIS GUIDELINES ===")
    context_parts.append(f"- Document Type: {doc_type.upper()}")
    context_parts.append("- Answer based ONLY on the document sections above")
    context_parts.append("- Cite specific sections, pages, or clause numbers")
    context_parts.append("- Use appropriate terminology for this document type")
    context_parts.append("- Be definitive when the document is clear")
    
    if doc_type == "legal":
        context_parts.append("- Focus on legal rights, obligations, and procedures")
        context_parts.append("- Include legal disclaimers where appropriate")
    elif doc_type == "insurance":
        context_parts.append("- Focus on coverage, benefits, and policy terms")
        context_parts.append("- Include relevant conditions and exclusions")
    elif doc_type == "technical":
        context_parts.append("- Focus on specifications, procedures, and technical requirements")
        context_parts.append("- Include safety and compliance information")
    
    return "\n".join(context_parts)

def extract_pdf_content(pdf_url: str) -> str:
    try:
        print(f"📥 Downloading PDF from: {pdf_url}")
        
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        content = ""
        for i, page in enumerate(pages):
            page_text = page.page_content.strip()
            if page_text:
                content += f"\n--- Page {i+1} ---\n{page_text}\n"
        
        content = content.replace('\n\n\n', '\n\n')
        content = content.replace('\t', ' ')
        
        os.unlink(temp_path)
        
        print(f"✅ PDF extracted successfully. Content length: {len(content)} characters")
        return content
        
    except Exception as e:
        print(f"❌ Error extracting PDF content: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF content: {str(e)}")

def process_documents(documents: Union[List[str], str]) -> str:
    if isinstance(documents, str):
        documents = [documents]
    
    processed_content = []
    
    for doc in documents:
        if is_url(doc):
            if doc.lower().endswith('.pdf') or 'pdf' in doc.lower():
                pdf_content = extract_pdf_content(doc)
                processed_content.append(pdf_content)
            else:
                raise HTTPException(status_code=400, detail=f"URL format not supported: {doc}")
        else:
            processed_content.append(doc)
    
    return "\n".join(processed_content)

def detect_document_type(content: str) -> str:
    """Detect document type based on content analysis"""
    content_lower = content.lower()
    
    # Insurance document indicators
    insurance_keywords = [
        'policy', 'coverage', 'premium', 'claim', 'deductible', 'beneficiary', 
        'insurer', 'insured', 'policyholder', 'insurance', 'underwriting',
        'actuary', 'benefits', 'copay', 'exclusions', 'preexisting condition'
    ]
    
    # Legal document indicators
    legal_keywords = [
        'contract', 'agreement', 'clause', 'party', 'parties', 'obligations', 
        'terms and conditions', 'hereby', 'pursuant to', 'jurisdiction',
        'liability', 'indemnity', 'termination', 'breach', 'litigation', 'statute'
    ]
    
    # Technical document indicators
    technical_keywords = [
        'specification', 'procedure', 'manual', 'technical', 'operation', 
        'maintenance', 'equipment', 'calibration', 'installation', 'troubleshooting',
        'diagnostic', 'algorithm', 'configuration', 'parameters', 'interface'
    ]
    
    # Academic document indicators
    academic_keywords = [
        'abstract', 'methodology', 'findings', 'conclusion', 'literature review',
        'hypothesis', 'research', 'study', 'data analysis', 'bibliography',
        'citation', 'peer-reviewed', 'journal', 'publication', 'dissertation'
    ]
    
    # Count occurrences of keywords for each type
    insurance_count = sum(1 for keyword in insurance_keywords if keyword in content_lower)
    legal_count = sum(1 for keyword in legal_keywords if keyword in content_lower)
    technical_count = sum(1 for keyword in technical_keywords if keyword in content_lower)
    academic_count = sum(1 for keyword in academic_keywords if keyword in content_lower)
    
    # Determine document type based on keyword counts
    max_count = max(insurance_count, legal_count, technical_count, academic_count)
    
    if max_count == 0:
        return "general"
    elif max_count == insurance_count:
        return "insurance"
    elif max_count == legal_count:
        return "legal"
    elif max_count == technical_count:
        return "technical"
    elif max_count == academic_count:
        return "academic"
    else:
        return "general"

def parse_llm_response(response_text: str) -> Dict:
    """Parse structured JSON response from LLM"""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # Fallback parsing for non-JSON responses
            return {
                "decision": "PENDING_REVIEW",
                "confidence_score": 0.5,
                "payout_amount": None,
                "reasoning": response_text,
                "policy_sections_referenced": [],
                "exclusions_applied": [],
                "coordination_of_benefits": {
                    "has_other_insurance": False,
                    "primary_insurance": None,
                    "secondary_insurance": None,
                    "primary_payment": None,
                    "remaining_amount": None
                },
                "processing_notes": ["Response parsing required fallback method"]
            }
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
        return {
            "decision": "PENDING_REVIEW",
            "confidence_score": 0.3,
            "payout_amount": None,
            "reasoning": f"Error parsing response: {response_text}",
            "policy_sections_referenced": [],
            "exclusions_applied": [],
            "coordination_of_benefits": {
                "has_other_insurance": False,
                "primary_insurance": None,
                "secondary_insurance": None,
                "primary_payment": None,
                "remaining_amount": None
            },
            "processing_notes": [f"JSON parsing error: {str(e)}"]
        }

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    received_token = credentials.credentials
    expected_token = os.getenv("AUTH_TOKEN")
    
    if not expected_token:
        raise HTTPException(status_code=500, detail="Server configuration error: AUTH_TOKEN not set.")

    if received_token == expected_token or received_token.strip() == expected_token.strip():
        return received_token
    
    raise HTTPException(status_code=403, detail="Invalid or expired token.")

# Initialize global variables (remove decision_engine)
vector_store = None
hybrid_retriever = None
processed_documents = []
batch_processor = None
last_request_model = None  # Track last used model

# Enhanced API Endpoints
@app.get("/")
def root():
    return {
        "message": "HackRx 6.0 Universal Document Analysis RAG Backend",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Universal document analysis (any document type)",
            "Specialized legal document processing",
            "Insurance policy analysis",
            "Technical document interpretation",
            "Smart document type detection",
            "Hybrid retrieval (Vector + BM25)",
            "Model rotation for rate limit avoidance",
            "Query logging and analytics"
        ],
        "supported_document_types": [
            "Legal documents (contracts, laws, regulations)",
            "Insurance policies and claims",
            "Technical manuals and specifications", 
            "Academic papers and reports",
            "Business documents and correspondence",
            "Government documents and constitutions",
            "Any text-based document"
        ],
        "supported_formats": ["text", "pdf_urls"],
        "endpoints": {
            "health": "/health",
            "run_query": "/hackrx/run",
            "debug_search": "/debug-search",
            "vector_stats": "/vector-stats",
            "llm_status": "/llm-status",
            "embeddings_status": "/embeddings-status",
            "model_rotation_stats": "/model-rotation-stats",
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
        "vector_store_ready": vector_store is not None,
        "hybrid_retriever_ready": hybrid_retriever is not None
    }

@app.post("/debug-search")
async def debug_search(request: DebugRequest):
    """Enhanced debug endpoint with hybrid retrieval information"""
    global hybrid_retriever
    
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No vector store available")
    
    try:
        if hybrid_retriever:
            docs = hybrid_retriever.retrieve_relevant_docs(request.question, k=6)
        else:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6, "fetch_k": 12}
            )
            docs = retriever.get_relevant_documents(request.question)
        
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
            "chunks": retrieved_chunks,
            "retrieval_method": "hybrid" if hybrid_retriever else "vector_only"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug search error: {str(e)}")

@app.get("/llm-status")
def llm_status():
    """Check LLM fallback manager status with rotation info"""
    global last_request_model
    
    if isinstance(llm, LLMFallbackManager):
        model_info = llm.get_current_model_info()
        return {
            "fallback_manager_active": True,
            "current_model": model_info,
            "rate_limited_models": list(llm.rate_limit_delays.keys()) if llm.rate_limit_delays else [],
            "available_models": [{"name": m["name"], "provider": m["provider"]} for m in llm.models],
            "total_requests_processed": llm.request_count,
            "smart_rotation_active": True,
            "last_request_model": last_request_model or "none"
        }
    else:
        return {
            "fallback_manager_active": False,
            "llm_available": llm is not None,
            "smart_rotation_active": False,
            "last_request_model": last_request_model or "none"
        }

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_enhanced_query(request: ClaimRequest, token: str = Depends(verify_token)):
    global vector_store, hybrid_retriever, processed_documents, batch_processor, last_request_model
    
    if embeddings is None or llm is None:
        raise HTTPException(status_code=500, detail="Core components not initialized")
    
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
    
    print(f"🎯 Processing {len(request.questions)} questions with SMART MODEL ROTATION - Request: {request_id}")
    
    # Check if we should skip processing for known documents
    should_skip = document_tracker.should_skip_processing(request.documents)
    
    # Detect document type for logging
    document_type = "unknown"
    
    try:
        # Step 1: Process documents only if not already processed
        if should_skip and vector_store is not None:
            print("⚡ CACHE HIT: Using existing embeddings for known documents")
            print("🚀 Skipping download and embedding - proceeding directly to Q&A")
            # Use existing vector store and processed documents
        else:
            print("📥 Processing documents (download + embedding needed)")
            
            # Process documents with chunking for large files
            document_content = process_documents(request.documents)
            
            # Detect document type
            document_type = detect_document_type(document_content)
            print(f"📋 Detected document type: {document_type}")
            
            if document_content.strip():
                docs = [Document(page_content=document_content)]
                
                # Smart chunking for large documents
                content_size = len(document_content)
                if content_size > 500000:  # Large document (500KB+)
                    print(f"📚 Large document detected ({content_size:,} chars), using optimized chunking")
                    text_splitter.chunk_size = 1200
                    text_splitter.chunk_overlap = 200
                    max_chunks = 100
                else:
                    text_splitter.chunk_size = 800
                    text_splitter.chunk_overlap = 300
                    max_chunks = 200
            
            chunks = text_splitter.split_documents(docs)
            
            if len(chunks) > max_chunks:
                print(f"⚡ Limiting chunks from {len(chunks)} to {max_chunks} for rate limit management")
                chunks = chunks[:max_chunks]
            
            print(f"📊 Created {len(chunks)} chunks from documents")
            
            print("🚀 Creating new vector store...")
            
            try:
                if len(chunks) > 50:
                    print("📦 Processing embeddings in batches")
                    batch_size = 20
                    all_embedded_chunks = []
                    
                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i:i+batch_size]
                        print(f"   🔤 Embedding batch {i//batch_size + 1}/{math.ceil(len(chunks)/batch_size)}")
                        
                        try:
                            batch_vector_store = FAISS.from_documents(batch_chunks, embeddings)
                            if i == 0:
                                vector_store = batch_vector_store
                            else:
                                vector_store.merge_from(batch_vector_store)
                            all_embedded_chunks.extend(batch_chunks)
                            
                            await asyncio.sleep(0.5)
                            
                        except Exception as e:
                            print(f"⚠️ Error in embedding batch {i//batch_size + 1}: {e}")
                            continue
                
                else:
                    print("🔤 Creating embeddings for small document")
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    processed_documents = chunks
                    print(f"✅ Embeddings completed: {len(chunks)} chunks embedded")
            
            except Exception as e:
                print(f"❌ Critical error in embedding creation: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")
            
            # Mark documents as processed
            document_tracker.mark_processed(request.documents)
            print("✅ Vector store created successfully - documents marked as processed")
            
            if hybrid_retriever is None:
                hybrid_retriever = HybridRetriever(vector_store, processed_documents)
                print("✅ Hybrid retriever initialized")
        
        if batch_processor is None:
            batch_processor = BatchProcessor(llm, ANSWER_PROMPT, max_batch_size=3)
            print("✅ Batch processor initialized")
        
        # Step 2: Process questions with intelligent model rotation and logging
        answers = []
        
        if vector_store is None:
            answers = ["No policy documents available for analysis."] * len(request.questions)
            
            # Log each failed question
            for i, question in enumerate(request.questions):
                query_logger.log_query(
                    request_id=f"{request_id}-q{i+1}",
                    question=question,
                    document_links=document_links,
                    document_type=document_type,
                    answer="No policy documents available for analysis.",
                    model_used=current_model_name,
                    processing_time=0,
                    chunks_retrieved=0,
                    success=False,
                    error_message="No documents provided"
                )
        else:
            question_batches = [request.questions[i:i+2] for i in range(0, len(request.questions), 2)]
            
            print(f"🔍 Processing {len(request.questions)} questions in {len(question_batches)} SMART batches")
            print("🎯 Each batch may use different models to avoid rate limits")
            
            question_index = 0
            
            for batch_idx, question_batch in enumerate(question_batches):
                print(f"   Processing batch {batch_idx + 1}/{len(question_batches)}")
                
                if batch_idx > 0 and isinstance(llm, LLMFallbackManager):
                    llm.rotate_to_next_available_model()
                    batch_model = llm.get_current_model_info()
                    print(f"   🔄 Batch {batch_idx + 1} using model: {batch_model.get('name', 'unknown')}")
                
                async def process_single_question(question):
                    nonlocal question_index
                    current_question_index = question_index
                    question_index += 1
                    
                    question_start_time = time.time()
                    chunks_retrieved = 0
                    
                    async with batch_processor.semaphore:
                        try:
                            if hybrid_retriever:
                                relevant_docs = hybrid_retriever.retrieve_relevant_docs(question, k=8)
                            else:
                                retriever = vector_store.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"k": 8, "fetch_k": 16}
                                )
                                relevant_docs = retriever.invoke(question)
                            
                            chunks_retrieved = len(relevant_docs)
                            
                            if relevant_docs:
                                context = create_optimized_context(relevant_docs, question, max_length=5000)
                                formatted_prompt = ANSWER_PROMPT.format(context=context, question=question)
                                
                                response = await llm.invoke_with_fallback(formatted_prompt)
                                
                                if hasattr(response, 'content'):
                                    response_text = response.content
                                else:
                                    response_text = str(response)
                                
                                cleaned_answer = batch_processor.clean_response(response_text)
                                
                                # Log successful question
                                question_time = time.time() - question_start_time
                                query_logger.log_query(
                                    request_id=f"{request_id}-q{current_question_index+1}",
                                    question=question,
                                    document_links=document_links,
                                    document_type=document_type,
                                    answer=cleaned_answer,
                                    model_used=current_model_name,
                                    processing_time=question_time,
                                    chunks_retrieved=chunks_retrieved,
                                    success=True,
                                    error_message=""
                                )
                                
                                return cleaned_answer
                            else:
                                answer = "No relevant policy information found for this question."
                                
                                # Log question with no relevant docs
                                question_time = time.time() - question_start_time
                                query_logger.log_query(
                                    request_id=f"{request_id}-q{current_question_index+1}",
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
                            print(f"Error processing question '{question[:50]}...': {str(e)}")
                            
                            # Log failed question
                            question_time = time.time() - question_start_time
                            query_logger.log_query(
                                request_id=f"{request_id}-q{current_question_index+1}",
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
                
                batch_start_time = time.time()
                tasks = [process_single_question(q) for q in question_batch]
                batch_answers = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(batch_answers):
                    if isinstance(result, Exception):
                        answers.append(f"Error processing question {len(answers)+1}: {str(result)}")
                    else:
                        answers.append(result)
                
                batch_time = time.time() - batch_start_time
                print(f"   Batch {batch_idx + 1} completed in {batch_time:.2f} seconds")
                
                if batch_idx < len(question_batches) - 1:
                    await asyncio.sleep(0.3)
        
        processing_time = time.time() - start_time
        avg_time_per_question = processing_time / len(request.questions)
        
        cache_status = "CACHE HIT" if should_skip else "CACHE MISS"
        print(f"🎉 {cache_status} processing completed in {processing_time:.2f} seconds")
        print(f"📊 Average time per question: {avg_time_per_question:.2f} seconds")
        print(f"🎯 Used model: {current_model_name}")
        print(f"📝 Logged {len(request.questions)} queries to CSV")
        
        if should_skip:
            print("⚡ Significant time saved by using existing embeddings!")
        
        return AnswerResponse(answers=answers)
        
    except Exception as e:
        print(f"❌ Error in smart rotation processing: {str(e)}")
        error_answers = [f"Error processing question: {str(e)}"] * len(request.questions)
        
        # Log batch error
        for i, question in enumerate(request.questions):
            query_logger.log_query(
                request_id=f"{request_id}-q{i+1}",
                question=question,
                document_links=document_links,
                document_type=document_type,
                answer="",
                model_used=current_model_name,
                processing_time=0,
                chunks_retrieved=0,
                success=False,
                error_message=str(e)
            )
        
        return AnswerResponse(answers=error_answers)

@app.get("/embeddings-status")
def embeddings_status():
    """Check embeddings fallback manager status"""
    if isinstance(embeddings, EmbeddingsFallbackManager):
        embedding_info = embeddings.get_current_embedding_info()
        return {
            "fallback_manager_active": True,
            "current_embedding": embedding_info,
            "rate_limited_embeddings": list(embeddings.rate_limit_delays.keys()) if embeddings.rate_limit_delays else [],
            "available_embeddings": [{"name": e["name"], "provider": e["provider"]} for e in embeddings.embeddings],
            "vector_store_ready": vector_store is not None,
            "reembedding_needed": False  # Embeddings are permanent once created
        }
    else:
        return {
            "fallback_manager_active": False,
            "embeddings_available": embeddings is not None,
            "vector_store_ready": vector_store is not None,
            "reembedding_needed": False
        }

@app.get("/vector-stats")
def vector_stats():
    """Get vector store statistics"""
    if vector_store is None:
        return {"status": "no_vector_store", "chunks": 0}
    
    return {
        "status": "ready",
        "total_chunks": len(processed_documents) if processed_documents else 0,
        "vector_store_type": "FAISS",
        "embeddings_permanent": True,
        "hybrid_retriever_active": hybrid_retriever is not None,
        "smart_rotation_active": True
    }

@app.get("/model-rotation-stats")
def model_rotation_stats():
    """Get model rotation statistics"""
    global last_request_model
    
    if isinstance(llm, LLMFallbackManager):
        return {
            "rotation_active": True,
            "total_models": len(llm.models),
            "current_model": llm.get_current_model_info(),
            "request_count": llm.request_count,
            "rate_limited_models": len(llm.rate_limit_delays),
            "last_request_model": last_request_model or "none",
            "available_models": [
                {
                    "name": m["name"], 
                    "provider": m["provider"],
                    "rate_limited": f"{m['provider']}:{m['name']}" in llm.rate_limit_delays
                } 
                for m in llm.models
            ]
        }
    else:
        return {
            "rotation_active": False,
            "last_request_model": last_request_model or "none"
        }

@app.post("/hackrx/structured", response_model=EnhancedAnswerResponse)
async def run_structured_query(request: ClaimRequest, token: str = Depends(verify_token)):
    """Structured endpoint for complex insurance claim decisions (optional)"""
    # This endpoint can be kept for specialized use cases but is not needed for basic Q&A
    raise HTTPException(status_code=501, detail="Structured decision engine not implemented in optimized version")

@app.get("/reset-rate-limits")
def reset_rate_limits(token: str = Depends(verify_token)):
    """Reset all rate limits for testing purposes"""
    global llm
    if isinstance(llm, LLMFallbackManager):
        llm.rate_limit_delays.clear()
        llm.rate_manager.request_counts.clear()
        llm.rate_manager.last_reset.clear()
        # Reset rate limit counters
        for provider in llm.rate_manager.rate_limits:
            llm.rate_manager.rate_limits[provider]['current_requests'] = 0
            llm.rate_manager.rate_limits[provider]['current_tokens'] = 0
        
        return {
            "status": "success",
            "message": "All rate limits have been reset",
            "available_models": len(llm.models),
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {"status": "no_rate_limits_to_reset"}

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

@app.get("/")
def root():
    return {
        "message": "HackRx 6.0 Universal Document Analysis RAG Backend",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Universal document analysis (any document type)",
            "Specialized legal document processing",
            "Insurance policy analysis",
            "Technical document interpretation",
            "Smart document type detection",
            "Hybrid retrieval (Vector + BM25)",
            "Model rotation for rate limit avoidance",
            "Query logging and analytics"
        ],
        "supported_document_types": [
            "Legal documents (contracts, laws, regulations)",
            "Insurance policies and claims",
            "Technical manuals and specifications", 
            "Academic papers and reports",
            "Business documents and correspondence",
            "Government documents and constitutions",
            "Any text-based document"
        ],
        "supported_formats": ["text", "pdf_urls"],
        "endpoints": {
            "health": "/health",
            "run_query": "/hackrx/run",
            "debug_search": "/debug-search",
            "vector_stats": "/vector-stats",
            "llm_status": "/llm-status",
            "embeddings_status": "/embeddings-status",
            "model_rotation_stats": "/model-rotation-stats",
            "query_stats": "/query-stats",
            "download_logs": "/download-logs"
        }
    }

class SimpleDocumentTracker:
    """Simple tracker to avoid re-downloading the same documents"""
    
    def __init__(self):
        self.processed_urls = set()
        
        # Pre-populate with known document links (normalized)
        self.known_document_links = {
            "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf",
            "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf", 
            "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf",
            "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf",
            "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf",
            "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
        }
        
        print(f"📋 Document tracker initialized with {len(self.known_document_links)} known documents")
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing query parameters"""
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
        return normalized
    
    def is_known_document(self, url: str) -> bool:
        """Check if URL is in the known document list"""
        if not is_url(url):
            return False
        normalized = self.normalize_url(url)
        return normalized in self.known_document_links
    
    def mark_processed(self, documents: Union[List[str], str]):
        """Mark documents as processed"""
        if isinstance(documents, str):
            documents = [documents]
        
        for doc in documents:
            if is_url(doc):
                normalized = self.normalize_url(doc)
                self.processed_urls.add(normalized)
                print(f"✅ Marked as processed: {normalized}")
    
    def was_processed_recently(self, documents: Union[List[str], str]) -> bool:
        """Check if documents were processed recently"""
        if isinstance(documents, str):
            documents = [documents]
        
        for doc in documents:
            if is_url(doc):
                normalized = self.normalize_url(doc)
                if normalized in self.processed_urls:
                    print(f"🔍 Document was processed recently: {normalized}")
                    return True
        
        return False
    
    def should_skip_processing(self, documents: Union[List[str], str]) -> bool:
        """Check if we should skip processing (known docs + already processed)"""
        if isinstance(documents, str):
            documents = [documents]
        
        # Check if all documents are known AND have been processed
        all_known = True
        all_processed = True
        
        for doc in documents:
            if is_url(doc):
                normalized = self.normalize_url(doc)
                if not self.is_known_document(doc):
                    all_known = False
                if normalized not in self.processed_urls:
                    all_processed = False
        
        should_skip = all_known and all_processed
        
        if should_skip:
            print("🎯 SKIPPING: All documents are known and already processed!")
            print("📚 Using existing embeddings for known documents")
        
        return should_skip
    
    def get_stats(self) -> Dict:
        """Get simple stats"""
        return {
            "processed_urls_count": len(self.processed_urls),
            "known_documents_count": len(self.known_document_links),
            "processed_urls": list(self.processed_urls),
            "known_documents": list(self.known_document_links)
        }

# Initialize simple document tracker
document_tracker = SimpleDocumentTracker()

@app.get("/tracker-stats")
def get_tracker_stats():
    """Get document tracker statistics"""
    return document_tracker.get_stats()

@app.get("/clear-tracker")
def clear_tracker():
    """Clear the document tracker"""
    count = len(document_tracker.processed_urls)
    document_tracker.processed_urls.clear()
    
    return {
        "status": "success", 
        "message": f"Cleared {count} processed URLs from tracker",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/test-tracker")
def test_tracker():
    """Test the document tracker with known URLs"""
    test_urls = [
        "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
        "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D"
    ]
    
    results = []
    for url in test_urls:
        is_known = document_tracker.is_known_document(url)
        was_processed = document_tracker.was_processed_recently([url])
        should_skip = document_tracker.should_skip_processing([url])
        
        results.append({
            "url": url[:80] + "...",
            "normalized": document_tracker.normalize_url(url),
            "is_known": is_known,
            "was_processed": was_processed, 
            "should_skip": should_skip
        })
    
    return {
        "test_results": results,
        "tracker_stats": document_tracker.get_stats()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting UNIVERSAL DOCUMENT ANALYSIS HackRx 6.0 RAG Backend...")
    print("📍 Server will be available at:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000") 
    print("🎯 Universal Document Features:")
    print("   - Analyzes ANY document type (legal, insurance, technical, etc.)")
    print("   - Smart document type detection and specialized responses")
    print("   - Legal document analysis with proper disclaimers")
    print("   - Insurance policy interpretation")
    print("   - Technical manual guidance")
    print("   - Intelligent model rotation (13 models)")
    print("   - Advanced rate limit management")
    print("   - Permanent embeddings (no re-embedding)")
    print("   - Hybrid retrieval (Vector + BM25)")
    print("   - Query logging and analytics (CSV)")
    print("🔧 Document Types Supported:")
    print("   - Legal: Contracts, laws, regulations, agreements")
    print("   - Insurance: Policies, claims, coverage documents")
    print("   - Technical: Manuals, specifications, procedures")
    print("   - Academic: Research papers, reports, studies")
    print("   - Business: Correspondence, proposals, reports")
    print("   - Government: Constitutions, statutes, regulations")
    print("   - General: Any text-based document")
    print("📊 Analytics Features:")
    print("   - Query logging to CSV file")
    print("   - Performance statistics")
    print("   - Model usage tracking")
    print("   - Success/failure analysis")
    
    uvicorn.run(app, host=HOST, port=PORT)
