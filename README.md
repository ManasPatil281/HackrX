# 📦 Hybrid RAG Document Query System  

## 📘 Project Overview  
This project implements a **FastAPI-based RAG (Retrieval-Augmented Generation) system** for document query processing. It combines vector search (via embeddings) and BM25 for hybrid document retrieval, supports multi-format document extraction, and includes fallback mechanisms for embeddings and LLMs.  

### ✅ Key Features  
- **Multi-format document support**: PDF, Word, PowerPoint, Excel, images, text, CSV, HTML, JSON.  
- **Hybrid retrieval**: Combines vector search and BM25 with relevance scoring.  
- **Fallback systems**: EmbeddingsFallbackManager and LLMFallbackManager ensure reliability.  
- **Caching**: DocumentCache stores processed content and embeddings.  
- **Rate limiting**: RateLimitManager controls API request frequency.  
- **Monitoring**: Health checks, query logs, and system statistics endpoints.  

### 🛠️ Tech Stack  
- **Framework**: FastAPI (with CORS and HTTPBearer auth)  
- **Document Processing**: `document_extractor` (PyPDF2, python-docx, PIL, etc.)  
- **Retrieval**: LangChain for vector search and BM25.  
- **Caching**: Custom `DocumentCache` (hash-based storage).  
- **Logging**: QueryLogger for audit and analytics.  

---

## 🧩 Architecture  
### 📁 Project Structure  
```  
repo_path/  
├── prompt.py  
├── batch_processor.py  
├── document_extractor.py  
├── rate_limit_manager.py  
├── tp.py  
├── doc_cache.py  
├── response_model.py  
├── llm_fallback_manager.py  
└── logger.py  
```  

### 🔗 Component Interactions  
1. **Document Processing**:  
   - `document_extractor.py` handles format detection and content extraction.  
   - Caching via `doc_cache.py` avoids redundant processing.  

2. **Query Workflow**:  
   - `tp.py` (FastAPI server) receives requests → `HybridRetriever` combines vector/BM25 → LLM generates answers.  

3. **Fallback Systems**:  
   - `llm_fallback_manager.py` and `EmbeddingsFallbackManager` switch models on failure.  

4. **Monitoring**:  
   - `/health`, `/rag-status`, and `/query-stats` endpoints expose system metrics.  

---

## ⚙️ Technical Details  
### 🧠 Core Components  
- **FastAPI Server**: CORS-enabled with HTTPBearer authentication.  
- **HybridRetriever**: Merges vector search and BM25 results.  
- **run_query()**: Executes full RAG workflow (retrieve → generate).  
- **document_extractor.py**: Format-agnostic content extraction pipeline.  

### 🛠️ Supporting Modules  
- **RateLimitManager**: Enforces request rate limits.  
- **DocumentCache**: Hash-based caching for documents and embeddings.  
- **QueryLogger**: Logs queries and responses to CSV.  

### 📝 Configuration  
- **Config File**: `config.py` (not summarized) likely handles API keys, model paths, and system settings.  

### 🌟 Additional Features  
- **Parallel Processing**: `BatchProcessor` for bulk queries.  
- **Debug Mode**: `/debug-search` endpoint for query expansion testing.  
- **Dynamic Fallback**: Automatic model switching for embeddings/LLM.  

---

## 🌐 API Reference  
### 📥 Endpoints  
| Method | Path              | Purpose                                  |  
|--------|-------------------|------------------------------------------|  
| `GET`  | `/health`         | System health check                      |  
| `GET`  | `/rag-status`    | RAG system configuration/status          |  
| `POST` | `/hackrx/run`     | Execute RAG query with caching/parallelism |  
| `GET`  | `/llm-status`     | Check active LLM and fallback status     |  
| `GET`  | `/embeddings-status` | View embedding model status         |  
| `GET`  | `/query-stats`    | Retrieve query logs and analytics      |  
| `GET`  | `/download-logs`  | Export query logs as CSV                 |  
| `POST` | `/clear-cache`    | Clear cached documents and embeddings    |  

### 🔄 Request/Response Formats  
- **POST /hackrx/run**  
  ```json  
  {  
    "query": "What is the main topic of the document?",  
    "pdf_url": "https://example.com/document.pdf"  
  }  
  ```  
  **Response**:  
  ```json  
  {  
    "answer": "The main topic is...",  
    "source_documents": ["page_12.pdf", "section_3.txt"]  
  }  
  ```  

---

## 🚀 Setup & Usage  
### 📥 Installation  
```bash  
git clone <repo-url>  
cd repo_path  
pip install -r requirements.txt  
```  

### ▶️ Run the Server  
```bash  
uvicorn main:app --reload  
```  

### 🧪 Example Usage  
```bash  
curl -X POST "http://localhost:8000/hackrx/run"  
-H "Authorization: Bearer <token>"  
-H "Content-Type: application/json"  
-d '{"query": "Summarize the report", "pdf_url": "https://example.com/report.pdf"}'  
```  

---  
*Documentation generated from codebase summaries. For advanced configuration, refer to `config.py`.*
