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

class LLMFallbackManager:
    """Manages multiple LLM models with automatic fallback and rate limit handling"""
    
    def __init__(self):
        self.models = []
        self.current_index = 0
        self.rate_limit_delays = {}  # Track models that hit rate limits
        
        # Add Mistral as PRIMARY LLM if API key is available (since Mixtral 8x7B is deprecated)
        if MISTRAL_API_KEY:
            try:
                from langchain_mistralai import ChatMistralAI
                # Mistral Large as primary (best performance)
                mistral_large = ChatMistralAI(
                    model="mistral-large-latest",
                    mistral_api_key=MISTRAL_API_KEY,
                    temperature=0.2,
                    max_tokens=2000
                )
                self.models.append({
                    "name": "mistral-large-latest",
                    "provider": "mistral",
                    "model": mistral_large
                })
                
                # Add Mistral Medium as backup
                mistral_medium = ChatMistralAI(
                    model="mistral-medium",
                    mistral_api_key=MISTRAL_API_KEY,
                    temperature=0.2,
                    max_tokens=2000
                )
                self.models.append({
                    "name": "mistral-medium",
                    "provider": "mistral",
                    "model": mistral_medium
                })
                
                print(f"✅ Mistral LLMs initialized successfully: {len(self.models)} models")
                print("🎯 Primary model: Mistral-Large-Latest (best for insurance analysis)")
            except Exception as e:
                print(f"⚠️ Error initializing Mistral LLM: {e}")
        
        # Add Groq LLMs as fallback if API key is available
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
                
                print(f"✅ Groq LLMs initialized successfully: added {len(self.models)} models total")
            except Exception as e:
                print(f"⚠️ Error initializing Groq LLM: {e}")
        
        if not self.models:
            print("❌ No LLM models could be initialized. Please check your API keys.")
        else:
            print("🔄 Model priority order:")
            for i, model in enumerate(self.models, 1):
                print(f"   {i}. {model['name']} ({model['provider']})")

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
        """Async version of invoke with fallback handling"""
        return await asyncio.to_thread(self.invoke, prompt)

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
    title="HackRx 6.0 Insurance RAG Backend", 
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
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

# Enhanced Insurance-Specific Prompt Template
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

# Enhanced Insurance-Specific Prompt Template for natural answers
INSURANCE_ANSWER_PROMPT = """
You are an expert insurance policy advisor analyzing the **National Parivar Mediclaim Plus Policy**. Your task is to provide precise, actionable answers based on the policy context.

POLICY CONTEXT:
{context}

QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. **DIRECT ANSWER FIRST**: Start with a clear YES/NO or definitive answer when possible
2. **POLICY REFERENCES**: Always cite specific section numbers and exact quotes
3. **COMPREHENSIVE SEARCH**: Look for related terms, synonyms, and indirect references
4. **PRACTICAL GUIDANCE**: Explain what the customer should do next
5. **CONCISE FORMAT**: Keep answers focused and avoid unnecessary repetition

ANSWER STRUCTURE:
1. **Direct Answer**: [Clear yes/no/depends answer]
2. **Policy Basis**: [Specific section and exact quote]
3. **Additional Details**: [Relevant conditions, limits, or procedures]
4. **Next Steps**: [What the customer should do]

IMPORTANT: 
- If information exists but uses different terminology, explain the connection
- Avoid saying "not explicitly mentioned" unless you've truly exhausted all possibilities
- Focus on what IS covered rather than what isn't, when possible
- Be definitive when the policy is clear

ANSWER:"""

# Create the enhanced prompt template
ANSWER_PROMPT = PromptTemplate(
    template=INSURANCE_ANSWER_PROMPT,
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
    """Handles batch processing of questions for efficiency"""
    
    def __init__(self, llm_manager, prompt_template, max_batch_size=5):
        self.llm_manager = llm_manager
        self.prompt_template = prompt_template
        self.max_batch_size = max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=10)  # Increased workers
    
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

# Initialize decision engine and global variables
decision_engine = InsuranceDecisionEngine()
vector_store = None
hybrid_retriever = None
processed_documents = []
batch_processor = None

# Helper functions (keeping existing ones and adding new)
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
    """Create enhanced context with better organization"""
    if not relevant_docs:
        return ""
    
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
    context_parts.append("=== POLICY SECTIONS ===\n")
    
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
                context_parts.append(f"\n--- SECTION {section_num} (Score: {score:.1f}) ---")
                context_parts.append(content)
            break
        
        context_parts.append(f"\n--- SECTION {section_num} (Score: {score:.1f}) ---")
        context_parts.append(content)
        current_length += len(content)
        section_num += 1
    
    context_parts.append(f"\n\n=== ANALYSIS NOTES ===")
    context_parts.append("- Answer based ONLY on the policy sections above")
    context_parts.append("- Cite specific section numbers and exact quotes")
    context_parts.append("- Be definitive when the policy is clear")
    
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

# Enhanced API Endpoints
@app.get("/")
def root():
    return {
        "message": "HackRx 6.0 Insurance RAG Backend with Decision Engine",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Insurance claim decision engine",
            "Coordination of benefits analysis",
            "Structured JSON responses",
            "Hybrid retrieval (Vector + BM25)",
            "Confidence scoring",
            "Audit trail support"
        ],
        "supported_formats": ["text", "pdf_urls"],
        "endpoints": {
            "health": "/health",
            "rag_status": "/rag-status", 
            "run_query": "/hackrx/run",
            "debug_search": "/debug-search",
            "vector_stats": "/vector-stats",
            "decision_engine_status": "/decision-engine-status",
            "embeddings_status": "/embeddings-status"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "embeddings_ready": embeddings is not None,
        "llm_ready": llm is not None,
        "vector_store_ready": vector_store is not None,
        "decision_engine_ready": decision_engine is not None,
        "hybrid_retriever_ready": hybrid_retriever is not None
    }

@app.get("/decision-engine-status")
def decision_engine_status():
    """Check decision engine configuration and rules"""
    return {
        "engine_active": True,
        "decision_rules": decision_engine.decision_rules,
        "supported_decisions": ["APPROVED", "DENIED", "PENDING_REVIEW"],
        "coordination_benefits_supported": True,
        "confidence_scoring_enabled": True
    }

@app.get("/embeddings-status")
def embeddings_status():
    """Check embeddings fallback manager status"""
    if isinstance(embeddings, EmbeddingsFallbackManager):
        embedding_info = embeddings.get_current_embedding_info()
        return {
            "fallback_manager_active": True,
            "current_embedding": embedding_info,
            "rate_limited_embeddings": list(embeddings.rate_limit_delays.keys()) if embeddings.rate_limit_delays else [],
            "available_embeddings": [{"name": e["name"], "provider": e["provider"]} for e in embeddings.embeddings]
        }
    else:
        return {
            "fallback_manager_active": False,
            "embeddings_available": embeddings is not None
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
        
        # Add decision engine analysis
        has_cob = decision_engine.detect_coordination_of_benefits(
            " ".join([doc.page_content for doc in docs]), 
            request.question
        )
        
        return {
            "question": request.question,
            "total_chunks_retrieved": len(docs),
            "chunks": retrieved_chunks,
            "decision_engine_analysis": {
                "coordination_of_benefits_detected": has_cob,
                "retrieval_method": "hybrid" if hybrid_retriever else "vector_only"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug search error: {str(e)}")

@app.get("/llm-status")
def llm_status():
    """Check LLM fallback manager status"""
    if isinstance(llm, LLMFallbackManager):
        model_info = llm.get_current_model_info()
        return {
            "fallback_manager_active": True,
            "current_model": model_info,
            "rate_limited_models": list(llm.rate_limit_delays.keys()) if llm.rate_limit_delays else [],
            "available_models": [{"name": m["name"], "provider": m["provider"]} for m in llm.models]
        }
    else:
        return {
            "fallback_manager_active": False,
            "llm_available": llm is not None
        }

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_enhanced_query(request: ClaimRequest, token: str = Depends(verify_token)):
    global vector_store, hybrid_retriever, processed_documents, batch_processor
    
    if embeddings is None or llm is None:
        raise HTTPException(status_code=500, detail="Core components not initialized")
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    print(f"🎯 Processing {len(request.questions)} questions with ENHANCED PARALLEL processing - Request: {request_id}")
    
    try:
        # Step 1: Process documents (if new)
        document_content = process_documents(request.documents)
        
        if document_content.strip():
            docs = [Document(page_content=document_content)]
            chunks = text_splitter.split_documents(docs)
            print(f"📊 Created {len(chunks)} chunks from documents")
            
            # Create or update vector store
            if vector_store is None:
                print("🚀 Creating new vector store...")
                vector_store = FAISS.from_documents(chunks, embeddings)
                processed_documents = chunks
                print("✅ Vector store created successfully")
            else:
                print("🔄 Adding documents to existing vector store...")
                vector_store.add_documents(chunks)
                processed_documents.extend(chunks)
                print("✅ Documents added to existing vector store")
            
            # Initialize hybrid retriever
            hybrid_retriever = HybridRetriever(vector_store, processed_documents)
            print("✅ Enhanced hybrid retriever initialized")
        
        # Initialize batch processor if not already done
        if batch_processor is None:
            batch_processor = BatchProcessor(llm, ANSWER_PROMPT, max_batch_size=10)
            print("✅ Batch processor initialized")
        
        # Step 2: ENHANCED processing with individual context per question
        answers = []
        
        if vector_store is None:
            answers = ["No policy documents available for analysis."] * len(request.questions)
        else:
            # Process each question with individual optimized context
            print(f"🔍 Processing {len(request.questions)} questions with individual context optimization")
            
            # Create tasks for parallel processing with individual contexts
            async def process_single_question(question):
                try:
                    if hybrid_retriever:
                        relevant_docs = hybrid_retriever.retrieve_relevant_docs(question, k=18)  # More docs per question
                    else:
                        retriever = vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 18, "fetch_k": 36}
                        )
                        relevant_docs = retriever.get_relevant_documents(question)
                    
                    if relevant_docs:
                        # Create question-specific enhanced context
                        context = create_enhanced_context(relevant_docs, question, max_length=10000)
                        
                        formatted_prompt = ANSWER_PROMPT.format(context=context, question=question)
                        
                        # Use fallback manager
                        if hasattr(llm, 'invoke_with_fallback'):
                            response = await llm.invoke_with_fallback(formatted_prompt)
                        else:
                            response = await asyncio.to_thread(llm.invoke, formatted_prompt)
                        
                        # Extract content from AIMessage object
                        if hasattr(response, 'content'):
                            response_text = response.content
                        else:
                            response_text = str(response)
                        
                        # Clean the response
                        cleaned_answer = batch_processor.clean_response(response_text) if batch_processor else response_text
                        return cleaned_answer
                    else:
                        return "No relevant policy information found for this question."
                        
                except Exception as e:
                    print(f"Error processing question '{question[:50]}...': {str(e)}")
                    return f"Error processing question: {str(e)}"
            
            # Process all questions in parallel with individual contexts
            batch_start_time = time.time()
            tasks = [process_single_question(q) for q in request.questions]
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            final_answers = []
            for i, result in enumerate(answers):
                if isinstance(result, Exception):
                    final_answers.append(f"Error processing question {i+1}: {str(result)}")
                else:
                    final_answers.append(result)
            
            answers = final_answers
            batch_time = time.time() - batch_start_time
            
            print(f"⚡ All {len(request.questions)} questions processed in {batch_time:.2f} seconds")
        
        processing_time = time.time() - start_time
        avg_time_per_question = processing_time / len(request.questions)
        
        print(f"🎉 TOTAL processing completed in {processing_time:.2f} seconds")
        print(f"📊 Average time per question: {avg_time_per_question:.2f} seconds")
        
        return AnswerResponse(answers=answers)
        
    except Exception as e:
        print(f"❌ Error in enhanced processing: {str(e)}")
        error_answers = [f"Error processing question: {str(e)}"] * len(request.questions)
        return AnswerResponse(answers=error_answers)

@app.post("/hackrx/structured", response_model=EnhancedAnswerResponse)
async def run_structured_query(request: ClaimRequest, token: str = Depends(verify_token)):
    """Enhanced endpoint that returns structured decisions with full metadata"""
    global vector_store, hybrid_retriever, processed_documents
    
    if embeddings is None or llm is None:
        raise HTTPException(status_code=500, detail="Core components not initialized")
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    audit_trail = [f"Request {request_id} started at {datetime.now().isoformat()}"]
    
    try:
        # Step 1: Process documents
        document_content = process_documents(request.documents)
        audit_trail.append("Documents processed successfully")
        
        if document_content.strip():
            docs = [Document(page_content=document_content)]
            chunks = text_splitter.split_documents(docs)
            print(f"Created {len(chunks)} chunks from documents")
            audit_trail.append(f"Created {len(chunks)} document chunks")
            
            # Create or update vector store
            if vector_store is None:
                print("Creating new vector store...")
                vector_store = FAISS.from_documents(chunks, embeddings)
                processed_documents = chunks
                print("✅ Vector store created successfully")
                audit_trail.append("New vector store created")
            else:
                print("Adding documents to existing vector store...")
                vector_store.add_documents(chunks)
                processed_documents.extend(chunks)
                audit_trail.append("Documents added to existing vector store")
            
            # Initialize hybrid retriever
            hybrid_retriever = HybridRetriever(vector_store, processed_documents)
            audit_trail.append("Hybrid retriever initialized")
        
        # Step 2: Process each question with enhanced decision logic
        decisions = []
        
        if vector_store is None:
            for question in request.questions:
                decisions.append(ClaimDecision(
                    question=question,
                    decision="PENDING_REVIEW",
                    confidence_score=0.0,
                    reasoning="No documents available for analysis",
                    processing_notes=["No vector store available"]
                ))
        else:
            for question in request.questions:
                try:
                    print(f"Processing question: {question}")
                    audit_trail.append(f"Processing question: {question[:50]}...")
                    
                    # Enhanced retrieval with multiple strategies
                    if hybrid_retriever:
                        relevant_docs = hybrid_retriever.retrieve_relevant_docs(question, k=8)
                    else:
                        retriever = vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 8, "fetch_k": 16}
                        )
                        relevant_docs = retriever.get_relevant_documents(question)
                    
                    if relevant_docs:
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        # Generate enhanced prompt with decision structure
                        formatted_prompt = ENHANCED_PROMPT.format(context=context, question=question)
                        
                        # Get LLM response using fallback manager
                        if hasattr(llm, 'invoke_with_fallback'):
                            response = await llm.invoke_with_fallback(formatted_prompt)
                        else:
                            response = llm.invoke(formatted_prompt)
                        
                        # Extract content from AIMessage object
                        if hasattr(response, 'content'):
                            response_text = response.content
                        else:
                            response_text = str(response)
                        
                        # Parse structured response
                        parsed_response = parse_llm_response(response_text)
                        
                        # Create decision object with enhanced data
                        decision = ClaimDecision(
                            question=question,
                            decision=parsed_response.get("decision", "PENDING_REVIEW"),
                            confidence_score=parsed_response.get("confidence_score", 0.5),
                            payout_amount=parsed_response.get("payout_amount"),
                            reasoning=parsed_response.get("reasoning", "Analysis completed"),
                            policy_sections_referenced=parsed_response.get("policy_sections_referenced", []),
                            exclusions_applied=parsed_response.get("exclusions_applied", []),
                            processing_notes=parsed_response.get("processing_notes", [])
                        )
                        
                        # Add coordination of benefits if detected
                        cob_data = parsed_response.get("coordination_of_benefits", {})
                        if cob_data and cob_data.get("has_other_insurance"):
                            decision.coordination_of_benefits = CoordinationOfBenefits(**cob_data)
                        
                        decisions.append(decision)
                        audit_trail.append(f"Decision generated: {decision.decision} (confidence: {decision.confidence_score})")
                    else:
                        decisions.append(ClaimDecision(
                            question=question,
                            decision="PENDING_REVIEW",
                            confidence_score=0.1,
                            reasoning="No relevant policy information found for this question",
                            processing_notes=["No relevant documents retrieved"]
                        ))
                        audit_trail.append("No relevant documents found")
                        
                except Exception as e:
                    print(f"❌ Error processing question '{question}': {str(e)}")
                    decisions.append(ClaimDecision(
                        question=question,
                        decision="PENDING_REVIEW",
                        confidence_score=0.0,
                        reasoning=f"Error during processing: {str(e)}",
                        processing_notes=[f"Processing error: {str(e)}"]
                    ))
                    audit_trail.append(f"Error processing question: {str(e)}")
        
        processing_time = time.time() - start_time
        audit_trail.append(f"Processing completed in {processing_time:.2f} seconds")
        
        # Create processing metadata
        current_model = llm.get_current_model_info()["name"] if hasattr(llm, 'get_current_model_info') else "unknown"
        metadata = ProcessingMetadata(
            request_id=request_id,
            processing_time=processing_time,
            chunks_analyzed=len(processed_documents) if processed_documents else 0,
            model_used=current_model,
            timestamp=datetime.now().isoformat()
        )
        
        return EnhancedAnswerResponse(
            decisions=decisions,
            processing_metadata=metadata,
            audit_trail=audit_trail
        )
        
    except Exception as e:
        print(f"❌ Error in enhanced RAG processing: {str(e)}")
        audit_trail.append(f"Fatal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced RAG Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced HackRx 6.0 RAG Backend Server with LLM Fallback...")
    print("📍 Server will be available at:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000") 
    print("🎯 HackRx 6.0 Features:")
    print("   - LLM Fallback Manager (Mistral + Groq)")
    print("   - Automatic rate limit handling with model switching")
    print("   - Multiple model variants for redundancy")
    print("   - LangChain batch processing for multiple questions")
    print("   - Simple answers format: {\"answers\": [\"...\"]}")
    print("   - Insurance policy analysis with section references")
    print("   - Hybrid retrieval (Vector + BM25)")
    print("   - Structured responses available at /hackrx/structured")
    print("   - Enhanced performance for multiple questions")
    print("   - Auto-reload enabled for development")
    
    uvicorn.run(app, host=HOST, port=PORT)
