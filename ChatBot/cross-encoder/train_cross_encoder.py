"""
train_cross_encoder.py — Train Cross-Encoder từ scratch

Sử dụng:
- Data: cross-encoder/data/train.jsonl + dev.jsonl
- FAISS: vector_data/legal_hf_cosine (để mine hard negatives)
- Output: cross-encoder/outputs/models/cross_encoder_v5fix/

Usage:
    python cross-encoder/train_cross_encoder.py
"""

import json
import random
import gc
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent

# Data paths - USE FAISS-MATCHED FORMAT
TRAIN_FILE = ROOT / "data" / "train_faiss.jsonl"
DEV_FILE = ROOT / "data" / "dev_faiss.jsonl"

# FAISS index for mining hard negatives
FAISS_INDEX = PROJECT_ROOT / "vector_data" / "legal_hf_cosine" / "index.faiss"
FAISS_META = PROJECT_ROOT / "vector_data" / "legal_hf_cosine" / "metadata.json"

# Bi-encoder for encoding queries (same as FAISS index)
BI_ENCODER = "Quockhanh05/Vietnam_legal_embeddings"

# Output - new model trained on FAISS-matched data
CE_OUTPUT = ROOT / "outputs" / "models" / "cross_encoder_faiss_matched"

# Training config
BASE_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_EPOCHS = 3  # Reduced for speed
CE_BATCH = 16
CE_MAX_LEN = 256
SEED = 42

# Mining config
TOP_MINE = 30
SKIP_TOP_K = 10  # Skip top-10 (too easy)
HARD_NEG_PER = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)

print(f"Device: {DEVICE}")
print(f"Train file: {TRAIN_FILE}")
print(f"FAISS index: {FAISS_INDEX}")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path, max_rows=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def is_same_passage(cand_text, pos_text, threshold=0.9):
    """Check if candidate is too similar to positive."""
    # Simple overlap check
    pos_words = set(pos_text.lower().split())
    cand_words = set(cand_text.lower().split())
    if not pos_words:
        return False
    overlap = len(pos_words & cand_words) / len(pos_words)
    return overlap > threshold


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    CE_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Load FAISS + Bi-encoder
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[1] Loading FAISS index and bi-encoder...")
    
    if not FAISS_INDEX.exists():
        print(f"Error: FAISS index not found at {FAISS_INDEX}")
        return
    
    faiss_index = faiss.read_index(str(FAISS_INDEX))
    print(f"  FAISS: {faiss_index.ntotal} vectors")
    
    with open(FAISS_META, "r", encoding="utf-8") as f:
        faiss_meta = json.load(f)
    print(f"  Metadata: {len(faiss_meta)} entries")
    
    bi_encoder = SentenceTransformer(BI_ENCODER, device=DEVICE)
    print(f"  Bi-encoder: {BI_ENCODER}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Load training data
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[2] Loading training data...")
    
    train_rows = load_jsonl(TRAIN_FILE)
    pos_train = [r for r in train_rows if r.get("label") == 1]
    random.shuffle(pos_train)
    print(f"  Train positives: {len(pos_train)}")
    
    dev_rows = load_jsonl(DEV_FILE)
    pos_dev = [r for r in dev_rows if r.get("label") == 1][:500]
    print(f"  Dev positives: {len(pos_dev)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Mine hard negatives
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[3] Mining hard negatives...")
    
    ce_train = []
    stats = {"pos": 0, "neg": 0, "no_neg": 0}
    
    for r in tqdm(pos_train, desc="Mining train negatives"):
        query = r.get("query", "").strip()
        pos_passage = r.get("passage", "").strip()
        
        if not query or not pos_passage:
            continue
        
        # Add positive
        ce_train.append({
            "query": query,
            "passage": pos_passage,
            "label": 1
        })
        stats["pos"] += 1
        
        # Encode and search
        q_emb = bi_encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
        
        scores, ids = faiss_index.search(q_emb, TOP_MINE)
        ids = ids[0].tolist()
        
        # Mine hard negatives (skip top SKIP_TOP_K)
        added = 0
        for fid in ids[SKIP_TOP_K:]:
            if fid < 0 or fid >= len(faiss_meta):
                continue
            if added >= HARD_NEG_PER:
                break
            
            cand = faiss_meta[fid]
            cand_text = cand.get("text", "")
            
            # Skip if too similar to positive
            if is_same_passage(cand_text, pos_passage):
                continue
            
            ce_train.append({
                "query": query,
                "passage": cand_text,
                "label": 0
            })
            added += 1
            stats["neg"] += 1
        
        if added == 0:
            stats["no_neg"] += 1
    
    print(f"  Positives: {stats['pos']}")
    print(f"  Negatives: {stats['neg']}")
    print(f"  No neg found: {stats['no_neg']}")
    print(f"  Total samples: {len(ce_train)}")
    
    # Build dev set
    ce_dev = []
    for r in tqdm(pos_dev[:200], desc="Mining dev negatives"):
        query = r.get("query", "").strip()
        pos_passage = r.get("passage", "").strip()
        
        if not query or not pos_passage:
            continue
        
        ce_dev.append({"query": query, "passage": pos_passage, "label": 1})
        
        q_emb = bi_encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
        
        scores, ids = faiss_index.search(q_emb, TOP_MINE)
        ids = ids[0].tolist()
        
        for fid in ids[SKIP_TOP_K:]:
            if fid < 0 or fid >= len(faiss_meta):
                continue
            cand = faiss_meta[fid]
            cand_text = cand.get("text", "")
            if not is_same_passage(cand_text, pos_passage):
                ce_dev.append({"query": query, "passage": cand_text, "label": 0})
                break
    
    print(f"  Dev samples: {len(ce_dev)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Train Cross-Encoder
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[4] Training Cross-Encoder...")
    
    # Free memory
    del bi_encoder, faiss_index
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Prepare samples
    random.shuffle(ce_train)
    train_samples = [
        InputExample(texts=[r["query"], r["passage"]], label=float(r["label"]))
        for r in ce_train
    ]
    dev_samples = [
        InputExample(texts=[r["query"], r["passage"]], label=float(r["label"]))
        for r in ce_dev
    ]
    
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Dev samples: {len(dev_samples)}")
    
    # Create cross-encoder
    ce_model = CrossEncoder(
        BASE_CE_MODEL,
        num_labels=1,
        max_length=CE_MAX_LEN,
        device=DEVICE
    )
    
    # Evaluator
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        dev_samples,
        name="dev"
    )
    
    warmup_steps = int(len(train_samples) / CE_BATCH * CE_EPOCHS * 0.1)
    
    print(f"  Base model: {BASE_CE_MODEL}")
    print(f"  Epochs: {CE_EPOCHS}")
    print(f"  Batch size: {CE_BATCH}")
    print(f"  Warmup steps: {warmup_steps}")
    
    t0 = time.time()
    
    ce_model.fit(
        train_dataloader=DataLoader(train_samples, shuffle=True, batch_size=CE_BATCH),
        evaluator=evaluator,
        epochs=CE_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(CE_OUTPUT),
        use_amp=(DEVICE == "cuda"),
        show_progress_bar=True
    )
    
    elapsed = (time.time() - t0) / 60
    print(f"\n  Training completed in {elapsed:.1f} minutes")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Save model
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[5] Saving model...")
    
    saved_path = CE_OUTPUT / "saved_model"
    ce_model.save(str(saved_path))
    
    # Save config
    config = {
        "base_model": BASE_CE_MODEL,
        "epochs": CE_EPOCHS,
        "batch_size": CE_BATCH,
        "max_length": CE_MAX_LEN,
        "skip_top_k": SKIP_TOP_K,
        "top_mine": TOP_MINE,
        "hard_neg_per": HARD_NEG_PER,
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "training_minutes": round(elapsed, 1)
    }
    
    with open(CE_OUTPUT / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to: {saved_path}")
    print("\n✓ Cross-encoder training complete!")


if __name__ == "__main__":
    main()
