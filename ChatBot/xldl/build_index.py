import os
import argparse
import numpy as np
import faiss

def build_index(vec_dir: str):
    emb_path = os.path.join(vec_dir, "embeddings.npy")
    idx_path = os.path.join(vec_dir, "index.faiss")

    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"❌ Không thấy embeddings.npy tại: {emb_path}")

    emb = np.load(emb_path).astype("float32")
    if emb.ndim != 2:
        raise ValueError(f"❌ embeddings.npy phải có shape (N, d). Hiện tại: {emb.shape}")

    n, dim = emb.shape

    # Vì bạn encode với normalize_embeddings=True => dùng cosine = inner product
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, idx_path)
    print(f"✅ Built index: {idx_path}")
    print(f"   N={index.ntotal}, dim={dim}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["legal_hf", "dek21", "phobert", "all"])
    args = ap.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base = os.path.abspath(os.path.join(BASE_DIR, "..", "vector_data"))
    print("📁 Using vector_data at:", base)

    if args.model == "all":
        for m in ["legal_hf", "dek21", "phobert"]:
            vec_dir = os.path.join(base, m)
            if os.path.isdir(vec_dir):
                build_index(vec_dir)
            else:
                print(f"⚠️ Skip (folder not found): {vec_dir}")
    else:
        vec_dir = os.path.join(base, args.model)
        if not os.path.isdir(vec_dir):
            raise FileNotFoundError(f"❌ Không thấy thư mục: {vec_dir}")
        build_index(vec_dir)

if __name__ == "__main__":
    main()
