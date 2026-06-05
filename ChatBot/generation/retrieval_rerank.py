"""
retrieval_rerank.py — Pipeline Retrieval + Rerank Standalone

Dựa trên notebook v6_hard_neg_ce.ipynb:
  Query → Bi-encoder encode → FAISS search (top-N) → CE rerank → Top-K

Models:
  - Bi-encoder:    outputs/models/bi_bge_m3_ft (V4 FT)
  - Cross-encoder: outputs/models/ce_bge_reranker_ft_v6 (V6 harder neg)
  - FAISS index:   notebooks/outputs/tmp/faiss_v4.index
  - Corpus:        built from data/train.jsonl, dev.jsonl, train_with_neg.jsonl
"""

import sys
import torch
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# ── Paths ────────────────────────────────────────────────────────────────────
GEN_DIR   = Path(__file__).parent                         # generation/
CHATBOT_ROOT = GEN_DIR.parent                             # ChatBot/
CE_ROOT   = CHATBOT_ROOT / "cross-encoder"                # cross-encoder/
NB_DIR    = CE_ROOT / "notebooks"                         # notebooks/
DATA_DIR  = CE_ROOT / "data"                              # data/
MDL_DIR   = CE_ROOT / "outputs" / "models"                # outputs/models/

DEFAULT_BI_MODEL  = MDL_DIR / "bi_bge_m3_ft"
DEFAULT_CE_MODEL  = MDL_DIR / "ce_bge_reranker_ft_v6"
DEFAULT_FAISS_IDX = CHATBOT_ROOT.parent / "vector_data" / "production" / "index.faiss"

# Thêm notebooks/ vào path để import pipeline_utils
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))


# ═════════════════════════════════════════════════════════════════════════════
# CORPUS BUILDER  (reuse logic từ pipeline_utils)
# ═════════════════════════════════════════════════════════════════════════════

def _load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file, skip malformed lines."""
    import json
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def build_corpus() -> List[Dict]:
    """
    Gom unique passages từ metadata.json trong production folder.
    Nếu không thấy, fallback sang train/dev/train_with_neg.
    Index trong list = FAISS id.
    """
    def _norm(t: str) -> str:
        import re
        if t is None: return ""
        # Remove whitespace, punctuation, and lowercase for robust matching
        return re.sub(r'[\W_]+', '', str(t)).lower()

    prod_meta = CHATBOT_ROOT.parent / "vector_data" / "production" / "metadata.json"

    if prod_meta.exists():
        import json
        print(f"[Corpus] Loading from production metadata: {prod_meta}")
        with open(prod_meta, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            
        corpus = []
        needs_enrichment = False
        for r in meta_data:
            m = r.get("metadata", {})
            source = m.get("source_file", "")
            if not source:
                needs_enrichment = True
            corpus.append({
                "passage":     r.get("text", ""),
                "chunk_index": m.get("chunk_index", -1),
                "van_ban":     m.get("van_ban",  ""),
                "chuong":      m.get("chuong",   "0"),
                "dieu":        m.get("dieu",     "0"),
                "khoan":       m.get("khoan",    "0"),
                "diem":        m.get("diem",     ""),
                "source_file": source,
                "page_number": m.get("page_number", 1),
            })
        
        if needs_enrichment:
            print(f"[Corpus] ⚠ Metadata incomplete, enriching from training data...")
            train_f = DATA_DIR / "train.jsonl"
            dev_f   = DATA_DIR / "dev.jsonl"
            
            enrich_map_text = {} # Match by normalized text
            enrich_map_key  = {} # Match by (van_ban, dieu, khoan, chunk_index)
            
            for path in [train_f, dev_f]:
                if path.exists():
                    for r in _load_jsonl(path):
                        p = r.get("passage", "")
                        meta = r.get("meta", {})
                        if p and meta.get("source_file"):
                            # Map 1: Text-based
                            enrich_map_text[_norm(p)] = meta
                            
                            # Map 2: Structure-based (more reliable if text varies slightly)
                            vb = meta.get("van_ban", "")
                            d  = meta.get("dieu", "")
                            k  = meta.get("khoan", "")
                            ci = meta.get("chunk_index", -1)
                            key = f"{_norm(vb)}|{_norm(d)}|{_norm(k)}|{ci}"
                            enrich_map_key[key] = meta
            
            # Patch corpus
            enriched_count = 0
            for item in corpus:
                if not item["source_file"]:
                    # Try Map 1 (Text-based)
                    norm_p = _norm(item["passage"])
                    match = enrich_map_text.get(norm_p)
                    
                    # Try Map 2 (Structure-based) if Map 1 failed
                    if not match:
                        key = f"{_norm(item['van_ban'])}|{_norm(item['dieu'])}|{_norm(item['khoan'])}|{item['chunk_index']}"
                        match = enrich_map_key.get(key)
                    
                    if match:
                        item["source_file"] = match.get("source_file", "")
                        item["page_number"] = match.get("page_number", 1)
                        enriched_count += 1
                    else:
                        # Map 3: Heuristic-based Guessing
                        # If van_ban contains "202/2025/QH15", it's the sapxepdonvihanhchinh decree
                        vb_lower = item["van_ban"].lower()
                        if "202" in vb_lower and ("2025" in vb_lower or "qh15" in vb_lower):
                            item["source_file"] = "Data/DataPhuc/NghiQuyet/NghiQuyetSapXepDonViHanhChinh/202qh.signed_sapxepdonvihanhchinh.pdf"
                        elif "303" in vb_lower and "cp" in vb_lower:
                            item["source_file"] = "Data/DataPhuc/303-cp.signed.pdf"
            print(f"[Corpus] Enriched {enriched_count} passages with source info ✓")

        print(f"[Corpus] {len(corpus)} passages ready")
        return corpus

    # Fallback logic
    train_f = DATA_DIR / "train.jsonl"
    dev_f   = DATA_DIR / "dev.jsonl"
    neg_f   = DATA_DIR / "train_with_neg.jsonl"

    seen = {}
    for path in [train_f, dev_f, neg_f]:
        if not path.exists():
            continue
        for r in _load_jsonl(path):
            p = r.get("passage", "")
            if p and p not in seen:
                meta = r.get("meta", {})
                seen[p] = {
                    "passage":     p,
                    "chunk_index": meta.get("chunk_index", -1),
                    "van_ban":     meta.get("van_ban",  ""),
                    "chuong":      meta.get("chuong",   ""),
                    "dieu":        meta.get("dieu",     ""),
                    "khoan":       meta.get("khoan",    ""),
                    "diem":        meta.get("diem",     ""),
                    "source_file": meta.get("source_file", ""),
                    "page_number": meta.get("page_number", 1),
                }
    corpus = list(seen.values())
    print(f"[Corpus] {len(corpus)} passages loaded from training JSONLs (fallback)")
    return corpus


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVAL + RERANK PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def _has_model_weights(model_dir: Path) -> bool:
    """
    Kiểm tra thư mục model có chứa file weights thật không.
    Trả False nếu thiếu hoặc chỉ là Git LFS pointer.
    """
    for name in ["model.safetensors", "pytorch_model.bin"]:
        p = model_dir / name
        if p.exists() and p.stat().st_size > 1000:
            return True
    return False

class RetrievalReranker:
    """
    Pipeline Retrieval + Rerank:
      1. Bi-encoder encode query
      2. FAISS search → top-N candidates
      3. Cross-encoder rerank candidates
      4. Return top-K results

    Sử dụng đúng model/index từ notebook v6.
    """

    def __init__(
        self,
        bi_model_path:  Optional[Path] = None,
        ce_model_path:  Optional[Path] = None,
        faiss_index_path: Optional[Path] = None,
        corpus:         Optional[List[Dict]] = None,
        device:         Optional[str] = None,
        ce_batch_size:  int = 64,
    ):
        from sentence_transformers import SentenceTransformer, CrossEncoder

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = (self.device == "cuda")
        self.ce_batch_size = ce_batch_size

        # ── Corpus ──
        self.corpus = corpus or build_corpus()

        # ── Bi-encoder ──
        bi_path = bi_model_path or DEFAULT_BI_MODEL
        if bi_path.exists() and _has_model_weights(bi_path):
            print(f"[RetrievalReranker] Loading bi-encoder (local): {bi_path.name}")
            self.bi_model = SentenceTransformer(
                str(bi_path), 
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16} if self.use_fp16 else {}
            )
        else:
            # Fallback: local FT thiếu weights (Git LFS chưa pull) → dùng base model
            fallback_bi = "BAAI/bge-m3"
            if bi_path.exists():
                print(f"[RetrievalReranker] ⚠ Local bi-encoder thiếu weights: {bi_path}")
                print(f"[RetrievalReranker]   (Chạy 'git lfs pull' trong thư mục model để fix)")
            else:
                print(f"[RetrievalReranker] ⚠ Bi-encoder not found: {bi_path}")
            print(f"[RetrievalReranker] → Fallback sang HuggingFace Hub: {fallback_bi}")
            self.bi_model = SentenceTransformer(
                fallback_bi, 
                device=self.device,
                model_kwargs={
                    "use_safetensors": True,
                    "torch_dtype": torch.float16 if self.use_fp16 else torch.float32
                }
            )

        # ── FAISS index ──
        faiss_path = faiss_index_path or DEFAULT_FAISS_IDX
        if faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
            print(f"[RetrievalReranker] FAISS index loaded: {self.index.ntotal} vectors")
        else:
            print(f"[RetrievalReranker] ⚠ FAISS index not found: {faiss_path}")
            print(f"[RetrievalReranker] → Building FAISS index from corpus ({len(self.corpus)} passages)...")
            self.index = self._build_faiss_index(faiss_path)

        # ── Cross-encoder ──
        ce_path = ce_model_path or DEFAULT_CE_MODEL
        if ce_path.exists() and _has_model_weights(ce_path):
            print(f"[RetrievalReranker] Loading cross-encoder (local): {ce_path.name}")
            self.ce_model = CrossEncoder(
                str(ce_path),
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16} if self.use_fp16 else {},
            )
        else:
            fallback_ce = "BAAI/bge-reranker-v2-m3"
            if ce_path.exists():
                print(f"[RetrievalReranker] ⚠ Local CE thiếu weights: {ce_path}")
            else:
                print(f"[RetrievalReranker] ⚠ Cross-encoder not found: {ce_path}")
            print(f"[RetrievalReranker] → Fallback sang HuggingFace Hub: {fallback_ce}")
            self.ce_model = CrossEncoder(
                fallback_ce,
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16} if self.use_fp16 else {},
            )

        print(f"[RetrievalReranker] Ready on {self.device} ✓")

    # ── Auto-build FAISS index ───────────────────────────────────────────

    def _build_faiss_index(self, save_path: Path):
        """Encode corpus bằng bi-encoder → build FAISS index → save."""
        from tqdm.auto import tqdm

        passages = [c["passage"] for c in self.corpus]
        print(f"[RetrievalReranker] Encoding {len(passages)} passages...")
        embs = self.bi_model.encode(
            passages, batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product (= cosine khi normalized)
        index.add(embs)

        # Save for reuse
        save_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_path))
        print(f"[RetrievalReranker] FAISS index built: {index.ntotal} vectors → saved to {save_path}")
        return index

    # ── Step 1: FAISS retrieval ──────────────────────────────────────────────

    def retrieve(self, query: str, top_n: int = 100) -> List[int]:
        """
        Encode query bằng bi-encoder → FAISS search → top-N candidate ids.
        """
        q_emb = self.bi_model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        _, I = self.index.search(q_emb, top_n)
        # Lọc id hợp lệ
        candidate_ids = [i for i in I[0].tolist() if 0 <= i < len(self.corpus)]
        return candidate_ids

    # ── Step 2: CE rerank ────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidate_ids: List[int],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Cross-encoder rerank candidates → trả về top-K results.

        Returns:
            List[dict] với keys: rank, ce_score, passage, van_ban, dieu, khoan,
                                 diem, chuong, chunk_index, faiss_id
        """
        if not candidate_ids:
            return []

        # Build (query, passage) pairs
        pairs = [(query, self.corpus[idx]["passage"]) for idx in candidate_ids]

        # CE predict
        scores = self.ce_model.predict(
            pairs,
            batch_size=self.ce_batch_size,
            show_progress_bar=False,
        )

        # Sort by score descending
        scored = sorted(
            zip(scores, candidate_ids),
            key=lambda x: x[0],
            reverse=True,
        )

        # Build result list
        results = []
        for rank, (score, idx) in enumerate(scored[:top_k], 1):
            m = self.corpus[idx]
            results.append({
                "rank":        rank,
                "ce_score":    round(float(score), 4),
                "passage":     m["passage"],
                "van_ban":     m.get("van_ban", ""),
                "chuong":      m.get("chuong", ""),
                "dieu":        m.get("dieu", ""),
                "khoan":       m.get("khoan", ""),
                "diem":        m.get("diem", ""),
                "chunk_index": m.get("chunk_index", -1),
                "source_file": m.get("source_file", ""),
                "page_number": m.get("page_number", 1),
                "faiss_id":    idx,
            })

        return results

    # ── Full pipeline ────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        top_n: int = 100,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Full pipeline:  query → retrieve (FAISS top-N) → rerank (CE) → top-K

        Args:
            query:  câu hỏi
            top_n:  số candidates từ FAISS (default 100)
            top_k:  số kết quả sau rerank (default 5)

        Returns:
            List[dict] top-K passages đã rerank, format:
            [{"rank": 1, "ce_score": 5.12, "passage": "...", ...}, ...]
        """
        candidate_ids = self.retrieve(query, top_n=top_n)
        results = self.rerank(query, candidate_ids, top_k=top_k)
        return results


# ═════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

# Module-level singleton để tránh load lại model mỗi lần gọi
_reranker_instance: Optional[RetrievalReranker] = None


def clear_reranker_cache():
    """Clear the module-level singleton to force reload next time."""
    global _reranker_instance
    _reranker_instance = None


def run_retrieval_rerank(
    query: str,
    top_k: int = 5,
    top_n: int = 100,
) -> List[Dict]:
    """
    Hàm tiện ích: load model (1 lần) → retrieve + rerank → top-K.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RetrievalReranker()
    return _reranker_instance.run(query, top_n=top_n, top_k=top_k)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval + Rerank Pipeline")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Câu query để tìm kiếm")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Số kết quả trả về (default: 5)")
    parser.add_argument("--top-n", "-n", type=int, default=100,
                        help="Số candidates từ FAISS (default: 100)")
    args = parser.parse_args()

    query = args.query
    if not query:
        query = input("❓ Nhập câu hỏi: ").strip()

    if not query:
        print("Chưa nhập query!")
        sys.exit(1)

    print(f"\n📝 Query: {query}")
    print(f"   top_n={args.top_n}, top_k={args.top_k}\n")

    results = run_retrieval_rerank(query, top_k=args.top_k, top_n=args.top_n)

    print(f"\n{'='*70}")
    print(f"  TOP-{args.top_k} KẾT QUẢ SAU RERANK")
    print(f"{'='*70}")
    for r in results:
        print(f"\n  [{r['rank']}] CE score = {r['ce_score']:.4f}")
        print(f"      Văn bản: {r['van_ban']}")
        if r['dieu']:
            cite = f"Điều {r['dieu']}"
            if r['khoan']:
                cite += f", Khoản {r['khoan']}"
            if r['diem']:
                cite += f", Điểm {r['diem']}"
            print(f"      Trích dẫn: {cite}")
        print(f"      Passage: {r['passage'][:150]}...")
    print(f"\n{'='*70}")
