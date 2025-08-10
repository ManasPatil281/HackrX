from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

from config import HF_TOKEN, MISTRAL_API_KEY


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
