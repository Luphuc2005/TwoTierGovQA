"""Quick test: login + create conversation + send chat message."""
import requests
import sys

BASE = "http://localhost:8080/api"

# 1. Health check
try:
    h = requests.get(f"{BASE}/health", timeout=5)
    print(f"[1] Health: {h.status_code} {h.json()}")
except Exception as e:
    print(f"[1] Health FAILED: {e}")
    sys.exit(1)

# 2. Register (ignore if exists)
r = requests.post(f"{BASE}/auth/register", json={
    "email": "apitest@test.com", "password": "test123", "full_name": "API Test"
}, timeout=10)
print(f"[2] Register: {r.status_code} {r.text[:200]}")

# 3. Login
login = requests.post(f"{BASE}/auth/login", json={
    "email": "apitest@test.com", "password": "test123"
}, timeout=10)
print(f"[3] Login: {login.status_code}")
if login.status_code != 200:
    print(f"    Response: {login.text[:200]}")
    sys.exit(1)

token = login.json().get("access_token")
headers = {"Authorization": f"Bearer {token}"}
print(f"[3] Token obtained: {token[:20]}...")

# 4. Create conversation  
conv = requests.post(f"{BASE}/conversations", json={"title": "Test"}, headers=headers, timeout=10)
print(f"[4] Create conversation: {conv.status_code} {conv.text[:200]}")
conv_id = conv.json().get("id")
print(f"[4] Conversation ID: {conv_id}")

# 5. Send chat
print(f"[5] Sending chat... (may take 30-60s on CPU)")
try:
    chat = requests.post(f"{BASE}/chat", json={
        "conversation_id": conv_id,
        "message": "xin chao"
    }, headers=headers, timeout=180)
    print(f"[5] Chat: {chat.status_code}")
    print(f"[5] Response: {chat.text[:1000]}")
except Exception as e:
    print(f"[5] Chat FAILED: {e}")
