import os, json, argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_MAP = {
    "legal_hf": "Quockhanh05/Vietnam_legal_embeddings",
    "dek21": "huyydangg/DEk21_hcmute_embedding",
    "phobert": "vinai/phobert-base",
}

def encode_phobert(query, device="cuda"):
    """Encode query using PhoBERT model from encoder_phoBERT.py"""
    from encoder_phoBERT import preprocess, split_and_embed

    # Preprocess with word tokenization
    preprocessed_query = preprocess(query)
    
    # Get embeddings using split_and_embed from encoder_phoBERT.py
    vectors, _ = split_and_embed(preprocessed_query, max_seq_length=256, overlap=32)
    
    # Return the first vector (or average if multiple chunks)
    if len(vectors) == 1:
        vec = vectors[0]
    else:
        # If query is split into multiple chunks, average them
        vec = np.mean(vectors, axis=0)
        # Re-normalize after averaging
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
    
    return vec.astype("float32").reshape(1, -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["legal_hf", "dek21", "phobert"])
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--rerank", action="store_true", help="Rerank top results with a Cross-Encoder")
    ap.add_argument("--rerank-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    ap.add_argument("--rerank-topk", type=int, default=20, help="Candidates to rerank (>= topk)")
    args = ap.parse_args()

    vec_dir = os.path.join("vector_data", args.model)
    index = faiss.read_index(os.path.join(vec_dir, "index.faiss"))
    with open(os.path.join(vec_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Encode query based on model type
    if args.model == "phobert":
        q = encode_phobert(args.query, device=args.device)
    else:
        encoder = SentenceTransformer(MODEL_MAP[args.model], device=args.device)
        q = encoder.encode([args.query], normalize_embeddings=True).astype("float32")

    search_k = args.topk
    if args.rerank:
        search_k = max(args.topk, args.rerank_topk)

    D, I = index.search(q, search_k)

    # Optional Cross-Encoder rerank
    if args.rerank:
        from sentence_transformers import CrossEncoder

        candidates = []
        candidate_idxs = I[0].tolist()
        candidate_scores = D[0].tolist()
        for idx, score in zip(candidate_idxs, candidate_scores):
            if idx < 0:
                continue
            item = meta[int(idx)]
            text = item.get("text", "")
            candidates.append((idx, score, text, item))

        rerank_k = min(args.rerank_topk, len(candidates))
        rerank_candidates = candidates[:rerank_k]
        pairs = [[args.query, c[2]] for c in rerank_candidates]

        cross_encoder = CrossEncoder(args.rerank_model, device=args.device)
        cross_scores = cross_encoder.predict(pairs)
        order = np.argsort(cross_scores)[::-1]

        reranked = []
        for rank in order:
            idx, faiss_score, text, item = rerank_candidates[int(rank)]
            reranked.append((idx, faiss_score, float(cross_scores[int(rank)]), item))

        # Keep remaining items (if any) in original FAISS order
        remaining = candidates[rerank_k:]
        reranked.extend([(idx, score, None, item) for idx, score, _, item in remaining])

        final = reranked[:args.topk]
    else:
        pairs = [(idx, score) for idx, score in zip(I[0].tolist(), D[0].tolist()) if idx >= 0]
        final = [(int(idx), float(score), None, meta[int(idx)]) for idx, score in pairs[:args.topk]]

    print(f"\nMODEL={args.model} | topk={args.topk}")
    print("QUERY:", args.query)
    for r, (idx, faiss_score, cross_score, item) in enumerate(final, 1):
        md = item.get("metadata", {})
        if cross_score is not None:
            print(f"\n#{r} faiss={faiss_score:.4f} | cross={cross_score:.4f}")
        else:
            print(f"\n#{r} score={faiss_score:.4f}")
        print(f"- {md.get('van_ban')} | Chương {md.get('chuong')} | Điều {md.get('dieu')} | Khoản {md.get('khoan')} | Điểm {md.get('diem')}")
        print(item.get("text","")[:240].replace("\n"," "), "...")
    print()

if __name__ == "__main__":
    main()
