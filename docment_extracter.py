import json
import os
import tempfile
from urllib.parse import urlparse
from fastapi import HTTPException
import requests

try:
    from langchain_community.document_loaders import (
        Docx2txtLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader,
        UnstructuredWordDocumentLoader, UnstructuredImageLoader,
        CSVLoader, UnstructuredHTMLLoader, TextLoader
    )
    MULTI_FORMAT_AVAILABLE = True
    print("✅ Multi-format document loaders available")
except ImportError as e:
    MULTI_FORMAT_AVAILABLE = False
    print(f"⚠️ Some document loaders not available: {e}")
    print("   To install: pip install python-docx openpyxl python-pptx pillow unstructured")

try:
    import easyocr
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
    print("✅ OCR libraries available for image processing")
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR libraries not available. Install with: pip install easyocr pytesseract pillow")

try:
    import docx
    import openpyxl
    from pptx import Presentation
    OFFICE_LIBS_AVAILABLE = True
    print("✅ Office document libraries available")
except ImportError:
    OFFICE_LIBS_AVAILABLE = False
    print("⚠️ Office libraries not available. Install with: pip install python-docx openpyxl python-pptx")


from doc_cache import DocumentCache
from langchain_community.document_loaders import PyPDFLoader
document_cache = DocumentCache()
# Helper function to detect document type from URL or filename
def detect_document_type(url_or_filename: str) -> str:
    """Detect document type from URL or filename"""
    url_lower = url_or_filename.lower()
    
    # Check for ZIP files first and mark as unsupported
    if any(ext in url_lower for ext in ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2']):
        return 'archive_unsupported'
    
    # PDF files
    if '.pdf' in url_lower:
        return 'pdf'
    
    # Word documents
    elif any(ext in url_lower for ext in ['.doc', '.docx']):
        return 'word'
    
    # PowerPoint presentations
    elif any(ext in url_lower for ext in ['.ppt', '.pptx']):
        return 'powerpoint'
    
    # Excel spreadsheets
    elif any(ext in url_lower for ext in ['.xls', '.xlsx', '.xlsm']):
        return 'excel'
    
    # Images
    elif any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']):
        return 'image'
    
    # Text files
    elif any(ext in url_lower for ext in ['.txt', '.md', '.rtf']):
        return 'text'
    
    # CSV files
    elif '.csv' in url_lower:
        return 'csv'
    
    # HTML files
    elif any(ext in url_lower for ext in ['.html', '.htm']):
        return 'html'
    
    # JSON files
    elif '.json' in url_lower:
        return 'json'
    
    # Default to PDF if contains 'pdf' anywhere or unknown
    elif 'pdf' in url_lower:
        return 'pdf'
    
    else:
        return 'unknown'

# Enhanced function to extract content from various document types
def extract_document_content(doc_url: str) -> str:
    """Extract content from various document formats"""
    # Check cache first
    cached_content = document_cache.get_cached_pdf_content(doc_url)  # Reuse PDF cache for all docs
    if cached_content:
        return cached_content
    
    doc_type = detect_document_type(doc_url)
    print(f"📄 Detected document type: {doc_type} for {doc_url[:50]}...")
    
    # Handle unsupported archive files
    if doc_type == 'archive_unsupported':
        error_msg = f"Archive files (.zip, .rar, .7z, .tar, .gz, .bz2) are not supported. Please extract the contents and provide direct links to individual documents."
        print(f"🚫 {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        print(f"📥 Downloading {doc_type.upper()} from: {doc_url}")
        
        # Download the document
        # Folder where you want to save
        # Folder where you want to save
        save_folder = "downloads"
        os.makedirs(save_folder, exist_ok=True)

        # Extract filename from URL path
        parsed_url = urlparse(doc_url)
        file_name = os.path.basename(parsed_url.path) or "downloaded_file"

        # Optional: Add timestamp to avoid overwriting
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(file_name)
        file_name = f"{name}_{timestamp}{ext}"

        # Full save path
        file_path = os.path.join(save_folder, file_name)

        # Download
        response = requests.get(doc_url, timeout=60)
        response.raise_for_status()

        # Save file
        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"File saved to: {file_path}")
        
        # Determine file extension
        if doc_type == 'pdf':
            suffix = '.pdf'
        elif doc_type == 'word':
            suffix = '.docx'
        elif doc_type == 'powerpoint':
            suffix = '.pptx'
        elif doc_type == 'excel':
            suffix = '.xlsx'
        elif doc_type == 'image':
            # Try to get actual extension from URL
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                if ext in doc_url.lower():
                    suffix = ext
                    break
            else:
                suffix = '.jpg'  # Default
        elif doc_type == 'text':
            suffix = '.txt'
        elif doc_type == 'csv':
            suffix = '.csv'
        elif doc_type == 'html':
            suffix = '.html'
        elif doc_type == 'json':
            suffix = '.json'
        else:
            suffix = '.pdf'  # Default fallback
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        # Extract content based on document type
        content = ""
        
        if doc_type == 'pdf':
            content = extract_pdf_content_from_file(temp_path)
            
        elif doc_type == 'word':
            content = extract_word_content(temp_path)
            
        elif doc_type == 'powerpoint':
            content = extract_powerpoint_content(temp_path)
            
        elif doc_type == 'excel':
            content = extract_excel_content(temp_path)
            
        elif doc_type == 'image':
            content = extract_image_content(temp_path)
            
        elif doc_type == 'text':
            content = extract_text_content(temp_path)
            
        elif doc_type == 'csv':
            content = extract_csv_content(temp_path)
            
        elif doc_type == 'html':
            content = extract_html_content(temp_path)
            
        elif doc_type == 'json':
            content = extract_json_content(temp_path)
            
        else:
            # Fallback to PDF extraction
            print(f"⚠️ Unknown document type {doc_type}, trying PDF extraction...")
            content = extract_pdf_content_from_file(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Cache the content
        document_cache.cache_pdf_content(doc_url, content)  # Reuse PDF cache method
        
        print(f"✅ {doc_type.upper()} extracted successfully. Content length: {len(content)} characters")
        return content
        
    except Exception as e:
        print(f"❌ Error extracting {doc_type.upper()} content: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract {doc_type.upper()} content: {str(e)}")

def extract_pdf_content_from_file(file_path: str) -> str:
    """Extract PDF content from file path"""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        content = ""
        for i, page in enumerate(pages):
            page_text = page.page_content.strip()
            if page_text:
                content += f"\n--- Page {i+1} ---\n{page_text}\n"
        
        # Clean up the content
        content = content.replace('\n\n\n', '\n\n')
        content = content.replace('\t', ' ')
        
        return content
    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")

def extract_word_content(file_path: str) -> str:
    """Extract content from Word documents"""
    try:
        if MULTI_FORMAT_AVAILABLE:
            # Try unstructured loader first
            try:
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                content = "\n\n".join([doc.page_content for doc in docs])
                return content
            except:
                pass
        
        if OFFICE_LIBS_AVAILABLE:
            # Fallback to python-docx
            doc = docx.Document(file_path)
            content = ""
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    content += f"{paragraph.text}\n"
            
            # Extract tables
            for table_num, table in enumerate(doc.tables):
                content += f"\n--- Table {table_num + 1} ---\n"
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    content += f"{row_text}\n"
            
            return content
        else:
            raise Exception("Word document libraries not available")
            
    except Exception as e:
        raise Exception(f"Word document extraction failed: {e}")

def extract_powerpoint_content(file_path: str) -> str:
    """Extract content from PowerPoint presentations"""
    try:
        if MULTI_FORMAT_AVAILABLE:
            # Try unstructured loader first
            try:
                loader = UnstructuredPowerPointLoader(file_path)
                docs = loader.load()
                content = "\n\n".join([doc.page_content for doc in docs])
                return content
            except:
                pass
        
        if OFFICE_LIBS_AVAILABLE:
            # Fallback to python-pptx
            prs = Presentation(file_path)
            content = ""
            
            for slide_num, slide in enumerate(prs.slides):
                content += f"\n--- Slide {slide_num + 1} ---\n"
                
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        content += f"{shape.text}\n"
                    
                    # Extract table content if present
                    if shape.shape_type == 19:  # Table
                        try:
                            table = shape.table
                            for row in table.rows:
                                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                                content += f"{row_text}\n"
                        except:
                            pass
            
            return content
        else:
            raise Exception("PowerPoint libraries not available")
            
    except Exception as e:
        raise Exception(f"PowerPoint extraction failed: {e}")

def extract_excel_content(file_path: str) -> str:
    """Extract content from Excel spreadsheets"""
    try:
        if MULTI_FORMAT_AVAILABLE:
            # Try unstructured loader first
            try:
                loader = UnstructuredExcelLoader(file_path)
                docs = loader.load()
                content = "\n\n".join([doc.page_content for doc in docs])
                return content
            except:
                pass
        
        if OFFICE_LIBS_AVAILABLE:
            # Fallback to openpyxl
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            content = ""
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                content += f"\n--- Sheet: {sheet_name} ---\n"
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                    if row_text.strip() and row_text != " | " * (len(row) - 1):
                        content += f"{row_text}\n"
            
            return content
        else:
            raise Exception("Excel libraries not available")
            
    except Exception as e:
        raise Exception(f"Excel extraction failed: {e}")

def extract_image_content(file_path: str) -> str:
    """Extract text from images using OCR"""
    try:
        if not OCR_AVAILABLE:
            return f"Image file detected but OCR not available. Install with: pip install easyocr pytesseract pillow"
        
        content = ""
        
        # Try EasyOCR first (often more accurate)
        try:
            reader = easyocr.Reader(['en'])  # English language
            results = reader.readtext(file_path)
            
            content += "--- OCR Text Extraction (EasyOCR) ---\n"
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Only include high-confidence text
                    content += f"{text}\n"
                    
        except Exception as e:
            print(f"EasyOCR failed: {e}, trying Tesseract...")
            
            # Fallback to Tesseract
            try:
                from PIL import Image
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                content += "--- OCR Text Extraction (Tesseract) ---\n"
                content += text
            except Exception as e2:
                content = f"OCR extraction failed with both EasyOCR and Tesseract: {e}, {e2}"
        
        return content
        
    except Exception as e:
        raise Exception(f"Image OCR extraction failed: {e}")

def extract_text_content(file_path: str) -> str:
    """Extract content from text files"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content
    except Exception as e:
        # Try different encodings
        for encoding in ['latin1', 'cp1252', 'ascii']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content
            except:
                continue
        raise Exception(f"Text file extraction failed: {e}")

def extract_csv_content(file_path: str) -> str:
    """Extract content from CSV files"""
    try:
        if MULTI_FORMAT_AVAILABLE:
            # Try unstructured CSV loader
            try:
                loader = CSVLoader(file_path)
                docs = loader.load()
                content = "\n\n".join([doc.page_content for doc in docs])
                return content
            except:
                pass
        
        # Fallback to pandas
        import pandas as pd
        df = pd.read_csv(file_path)
        
        content = "--- CSV Data ---\n"
        content += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Convert to readable format
        for index, row in df.iterrows():
            row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            content += f"Row {index + 1}: {row_text}\n"
            
            # Limit to first 100 rows for large files
            if index >= 99:
                content += f"\n... (showing first 100 rows of {len(df)} total)\n"
                break
        
        return content
        
    except Exception as e:
        raise Exception(f"CSV extraction failed: {e}")

def extract_html_content(file_path: str) -> str:
    """Extract content from HTML files"""
    try:
        if MULTI_FORMAT_AVAILABLE:
            # Try unstructured HTML loader
            try:
                loader = UnstructuredHTMLLoader(file_path)
                docs = loader.load()
                content = "\n\n".join([doc.page_content for doc in docs])
                return content
            except:
                pass
        
        # Fallback to BeautifulSoup if available
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            content = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = '\n'.join(chunk for chunk in chunks if chunk)
            
            return content
            
        except ImportError:
            # Simple HTML tag removal
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            import re
            # Remove HTML tags
            clean_text = re.sub('<.*?>', '', html_content)
            # Clean up whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            return clean_text
            
    except Exception as e:
        raise Exception(f"HTML extraction failed: {e}")

def extract_json_content(file_path: str) -> str:
    """Extract content from JSON files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def json_to_text(obj, level=0):
            """Convert JSON object to readable text"""
            indent = "  " * level
            text = ""
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    text += f"{indent}{key}: "
                    if isinstance(value, (dict, list)):
                        text += "\n" + json_to_text(value, level + 1)
                    else:
                        text += f"{value}\n"
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    text += f"{indent}[{i}]: "
                    if isinstance(item, (dict, list)):
                        text += "\n" + json_to_text(item, level + 1)
                    else:
                        text += f"{item}\n"
            else:
                text += f"{indent}{obj}\n"
            
            return text
        
        content = "--- JSON Data ---\n"
        content += json_to_text(data)
        
        return content
        
    except Exception as e:
        raise Exception(f"JSON extraction failed: {e}")

# Update the original extract_pdf_content function to use the new multi-format function
def extract_pdf_content(pdf_url: str) -> str:
    """Legacy function - redirects to multi-format extraction"""
    return extract_document_content(pdf_url)