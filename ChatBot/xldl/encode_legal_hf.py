import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
CHUNKS_PATH = "output_nghidinh/chunks_clean.json"
OUT_DIR = os.path.join("vector_data", "legal_hf_cosine")
MODEL_NAME = "Quockhanh05/Vietnam_legal_embeddings"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# LOAD ENCODER 
encoder = SentenceTransformer(MODEL_NAME, device="cuda")

# =========================
# ENCODE (COSINE MODE)
# =========================
embeddings = encoder.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

# 🔥 BẮT BUỘC normalize để dùng cosine
faiss.normalize_L2(embeddings)

print("Embeddings shape:", embeddings.shape)

# =========================
# BUILD COSINE INDEX
# =========================
dim = embeddings.shape[1]

# 👇 IP + normalized = cosine similarity
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# =========================
# SAVE
# =========================
faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)

with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("✅ COSINE index built successfully")
print("Metric type:", index.metric_type)  # phải = 1
print("Total vectors:", index.ntotal)