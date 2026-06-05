import os
import sys

os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Test with MT5 and Vit5
TOKENIZER_CANDIDATES = [
    "google/mt5-base", 
    "google/mt5-small", 
    "VietAI/vit5-base"
]

MODEL_NAME = "protonx-models/protonx-legal-tc"
print(f"Loading Model: {MODEL_NAME}")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir='D:/huggingface_cache')

test_sentence = "ngui ky: Cong thong tin chinh phu"

for tok_name in TOKENIZER_CANDIDATES:
    print(f"\n--- Testing Tokenizer: {tok_name} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name, cache_dir='D:/huggingface_cache')
        inputs = tokenizer(test_sentence, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
            
        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"RESULT: {corrected}")
    except Exception as e:
        print(f"FAILED: {e}")
