import asyncio
import time
from config import MISTRAL_API_KEY
from rate_limit_manager import RateLimitManager
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
                    # "mistral-saba-2502",      # New specialized model
                ]
                
                for model_name in mistral_models:
                    try:
                        mistral_model = ChatMistralAI(
                            model=model_name,
                            mistral_api_key=MISTRAL_API_KEY,
                            temperature=0.3,
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
    
    def generate_answer_direct(self, messages):
        """
        Send a list of messages (OpenAI-style) directly to the underlying LLM.
        Assumes self.llm is ChatCompletion-like.
        """
        try:
            resp = self.llm.invoke(messages)  # adjust if you're using different API
            # Return dict with 'content' key for compatibility
            return {"content": resp.content if hasattr(resp, "content") else str(resp)}
        except Exception as e:
            raise RuntimeError(f"LLM direct call failed: {e}")


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
