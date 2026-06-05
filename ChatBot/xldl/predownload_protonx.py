import os
import sys

# Đổi cache HuggingFace sang ổ D
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    MODEL_NAME = "protonx-models/protonx-legal-tc"
    print(f"Pre-downloading {MODEL_NAME} to D:/huggingface_cache. This might take a while...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir='D:/huggingface_cache')
    print("Tokenizer downloaded successfully.")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir='D:/huggingface_cache')
    print("Model downloaded successfully!")
    
except Exception as e:
    print(f"Error downloading: {e}")
