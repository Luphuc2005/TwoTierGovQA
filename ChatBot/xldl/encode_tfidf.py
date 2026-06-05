import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ===== CONFIG =====
CHUNKS_PATH = "output_nghidinh/chunks_clean.json"

OUT_DIR = os.path.join("vector_data", "tfidf")   # thư mục output riêng


os.makedirs(OUT_DIR, exist_ok=True)

# ===== LOAD DATA =====
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# ===== TF-IDF VECTORIZER =====
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2,
    sublinear_tf=True,
    norm="l2"
)

tfidf_matrix = vectorizer.fit_transform(texts)

# ===== SAVE =====
with open(os.path.join(OUT_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# sparse matrix
import scipy.sparse as sp
sp.save_npz(os.path.join(OUT_DIR, "tfidf_matrix.npz"), tfidf_matrix)

print("✅ TF-IDF built")
print("   Shape:", tfidf_matrix.shape)
print("   Vocabulary size:", len(vectorizer.vocabulary_))
