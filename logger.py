import csv
from datetime import datetime
from pathlib import Path

import pandas as pd


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
