# API Keys
import os


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))