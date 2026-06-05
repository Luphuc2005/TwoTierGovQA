import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ===== CONFIG =====
CHUNKS_PATH = r"D:\GitHub\ChatBot\output_nghidinh\chunks_clean.json"
OUT_DIR = os.path.join("vector_data", "dek21")   # thư mục output riêng
MODEL_NAME = "huyydangg/DEk21_hcmute_embedding"

# Nếu bạn muốn tự tạo thư mục thì bật dòng này:
os.makedirs(OUT_DIR, exist_ok=True)

# ===== LOAD DATA =====
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# ===== LOAD ENCODER =====
encoder = SentenceTransformer(
    MODEL_NAME,
    device="cuda"  # nếu không có GPU: đổi thành "cpu"
)

# ===== ENCODE =====
embeddings = encoder.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True   # để dùng cosine với IndexFlatIP
).astype("float32")

# ===== SAVE =====
np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)

with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("✅ Done encoding:", embeddings.shape)
print("📁 Saved to:", os.path.abspath(OUT_DIR))
