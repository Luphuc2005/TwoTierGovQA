import os
import sys
import traceback

os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'

try:
    print("Testing transformers import...")
    import torch
    from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {DEVICE}")
    
    MODEL_NAME = "protonx-models/protonx-legal-tc"
    print(f"Loading {MODEL_NAME} from D:/huggingface_cache ...")
    
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir='D:/huggingface_cache')
    print("Tokenizer loaded successfully.")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir='D:/huggingface_cache')
    model.to(DEVICE)
    model.eval()
    print("Model loaded and moved to device successfully!")
    
except Exception as e:
    print("FAILED TO LOAD PROTONX MODEL. FULL TRACEBACK:")
    traceback.print_exc()
