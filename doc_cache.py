import hashlib
from pathlib import Path
import pickle


class DocumentCache:
    """Caches PDF content, chunks, embeddings, and vector stores to avoid repeated processing"""
    
    def __init__(self, cache_dir="./cache1"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.pdf_cache = {}
        self.chunks_cache = {}
        self.vector_store_cache = {}
        self.embedding_cache = {}
        print(f"✅ Document cache initialized at {self.cache_dir}")
    
    def get_url_hash(self, url):
        """Generate a hash for PDF URL"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def get_content_hash(self, content):
        """Generate a hash for document content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_cached_pdf_content(self, pdf_url):
        """Get cached PDF content if available"""
        url_hash = self.get_url_hash(pdf_url)
        cache_file = self.cache_dir / f"pdf_{url_hash}.txt"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"📁 Using cached PDF content for {pdf_url[:50]}...")
                return content
            except Exception as e:
                print(f"⚠️ Error reading PDF cache: {e}")
        
        return None
    
    def cache_pdf_content(self, pdf_url, content):
        """Cache PDF content to avoid repeated downloads"""
        try:
            url_hash = self.get_url_hash(pdf_url)
            cache_file = self.cache_dir / f"pdf_{url_hash}.txt"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"💾 Cached PDF content for {pdf_url[:50]}")
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
    
    def get_cached_embeddings(self, content_hash):
        """Get cached embeddings if available"""
        embeddings_file = self.cache_dir / f"embeddings_{content_hash}.pkl"
        
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'rb') as f:
                    embeddings_data = pickle.load(f)
                print(f"📁 Using cached embeddings for content hash {content_hash[:8]}...")
                return embeddings_data
            except Exception as e:
                print(f"⚠️ Error reading embeddings cache: {e}")
        
        return None
    
    def cache_embeddings(self, content_hash, embeddings_data):
        """Cache embeddings to avoid repeated computation"""
        try:
            embeddings_file = self.cache_dir / f"embeddings_{content_hash}.pkl"
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            print(f"💾 Cached embeddings for content hash {content_hash[:8]}")
            return True
        except Exception as e:
            print(f"⚠️ Error caching embeddings: {e}")
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
    
    def get_cached_complete_data(self, pdf_urls):
        """Get complete cached data (content, chunks, vector store) for PDF URLs"""
        # Create a combined hash for all URLs
        combined_content = ""
        
        for url in pdf_urls:
            cached_content = self.get_cached_pdf_content(url)
            if cached_content is None:
                return None  # If any PDF is not cached, return None
            combined_content += cached_content + "\n\n"
        
        if not combined_content.strip():
            return None
        
        content_hash = self.get_content_hash(combined_content)
        
        # Check if we have all cached components
        cached_chunks = self.get_cached_chunks(content_hash)
        cached_vector_store = self.get_cached_vector_store(content_hash)
        
        if cached_chunks and cached_vector_store:
            return {
                "content": combined_content,
                "content_hash": content_hash,
                "chunks": cached_chunks,
                "vector_store": cached_vector_store
            }
        
        return None
    
    def cache_complete_data(self, pdf_urls, content, chunks, vector_store):
        """Cache complete data for PDF URLs"""
        content_hash = self.get_content_hash(content)
        
        # Cache individual PDF contents
        if isinstance(pdf_urls, list):
            for url in pdf_urls:
                if not self.get_cached_pdf_content(url):
                    # For simplicity, cache the full content for each URL
                    # In production, you might want to cache individual PDF contents
                    self.cache_pdf_content(url, content)
        
        # Cache chunks and vector store
        self.cache_chunks(content_hash, chunks)
        self.cache_vector_store(content_hash, vector_store)
        
        return content_hash
    
    def get_cache_stats(self):
        """Get statistics about the cache"""
        try:
            pdf_files = list(self.cache_dir.glob("pdf_*.txt"))
            chunks_files = list(self.cache_dir.glob("chunks_*.pkl"))
            vector_files = list(self.cache_dir.glob("vector_*.pkl"))
            embedding_files = list(self.cache_dir.glob("embeddings_*.pkl"))
            
            # Calculate total size
            pdf_size = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)  # MB
            chunks_size = sum(f.stat().st_size for f in chunks_files) / (1024 * 1024)  # MB
            vector_size = sum(f.stat().st_size for f in vector_files) / (1024 * 1024)  # MB
            embedding_size = sum(f.stat().st_size for f in embedding_files) / (1024 * 1024)  # MB
            
            return {
                "pdf_files": len(pdf_files),
                "chunks_files": len(chunks_files),
                "vector_files": len(vector_files),
                "embedding_files": len(embedding_files),
                "pdf_size_mb": round(pdf_size, 2),
                "chunks_size_mb": round(chunks_size, 2),
                "vector_size_mb": round(vector_size, 2),
                "embedding_size_mb": round(embedding_size, 2),
                "total_size_mb": round(pdf_size + chunks_size + vector_size + embedding_size, 2),
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
            self.embedding_cache = {}
            
            print("🧹 Cache cleared successfully")
            return True
        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")
            return False
