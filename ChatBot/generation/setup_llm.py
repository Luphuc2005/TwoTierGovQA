"""
setup_llm.py — Setup LLM API keys

Hướng dẫn lấy API key miễn phí:

1. Google Gemini (Free):
   - Vào https://aistudio.google.com/app/apikey
   - Click "Create API Key"
   - Copy key và set environment variable

2. OpenRouter — Qwen3 (Free tier):
   - Vào https://openrouter.ai/keys
   - Đăng ký và tạo API key
   - Free models: qwen/qwen3-30b-a3b:free, qwen/qwen3-8b:free
   - Paid: qwen/qwen3-235b-a22b

Fallback chain: Gemini → Qwen3 (OpenRouter) → Local placeholder

Usage:
    # Windows PowerShell — Gemini (primary)
    $env:GEMINI_API_KEY = "your-key-here"
    
    # OpenRouter — Qwen3 (fallback khi Gemini hết quota)
    $env:OPENROUTER_API_KEY = "sk-or-v1-..."
    
    # Chạy với Gemini + auto-fallback Qwen3
    python cross-encoder/generation/run_generation.py --backend gemini --query "..." -v
    
    # Chạy trực tiếp với Qwen3
    python cross-encoder/generation/run_generation.py --backend qwen --query "..." -v
    
    # Chạy với Qwen3 model cụ thể
    python cross-encoder/generation/run_generation.py --backend qwen --model "qwen/qwen3-235b-a22b" --query "..."
"""

import os
import sys

def check_api_keys():
    """Check which API keys are available."""
    
    print("\n" + "=" * 60)
    print("LLM API Keys Status")
    print("=" * 60)
    
    keys = {
        "GEMINI_API_KEY": "Google Gemini (gemini-2.0-flash) — Primary",
        "OPENROUTER_API_KEY": "OpenRouter / Qwen3 (qwen3-30b-a3b:free) — Fallback",
        "OPENAI_API_KEY": "OpenAI (GPT-4o-mini)",
    }
    
    found = False
    for key, desc in keys.items():
        value = os.environ.get(key)
        if value:
            masked = value[:10] + "..." + value[-4:] if len(value) > 14 else "***"
            print(f"✓ {key}: {masked}")
            print(f"  → {desc}")
            found = True
        else:
            print(f"✗ {key}: Not set")
    
    if not found:
        print("\n⚠ Không có API key nào được cài đặt.")
        print("\nCách cài đặt:")
        print("")
        print("  # Gemini (miễn phí) — Primary:")
        print('  $env:GEMINI_API_KEY = "AIza..."')
        print("")
        print("  # OpenRouter / Qwen3 (miễn phí) — Fallback:")
        print('  $env:OPENROUTER_API_KEY = "sk-or-v1-..."')
    
    return found


def test_gemini():
    """Test Gemini API."""
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set")
        return False
    
    import urllib.request
    import json
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": "Xin chào! Trả lời ngắn gọn."}]}],
        "generationConfig": {"maxOutputTokens": 50}
    }).encode()
    
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json"
    })
    
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
            print(f"✓ Gemini OK: {text[:100]}...")
            return True
    except Exception as e:
        print(f"✗ Gemini Error: {e}")
        return False


def test_openrouter():
    """Test OpenRouter API (Qwen3)."""
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return False
    
    import urllib.request
    import json
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    body = json.dumps({
        "model": "qwen/qwen3-30b-a3b:free",
        "messages": [{"role": "user", "content": "Xin chào! Trả lời ngắn gọn bằng tiếng Việt."}],
        "max_tokens": 50
    }).encode()
    
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/legal-rag",
    })
    
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
            text = resp["choices"][0]["message"]["content"]
            print(f"✓ OpenRouter OK: {text[:100]}...")
            return True
    except Exception as e:
        print(f"✗ OpenRouter Error: {e}")
        return False


if __name__ == "__main__":
    found = check_api_keys()
    
    if found:
        print("\n" + "-" * 40)
        print("Testing APIs...")
        print("-" * 40)
        
        if os.environ.get("GEMINI_API_KEY"):
            test_gemini()
        if os.environ.get("OPENROUTER_API_KEY"):
            test_openrouter()
