"""
pipeline_utils.py — Shared utilities for cross-encoder/notebooks/
"""

import json
import csv
from pathlib import Path
from tqdm.auto import tqdm

# ── Paths ──────────────────────────────────────────
ROOT     = Path(__file__).parent          # cross-encoder/notebooks/
BASE_DIR = ROOT.parent                    # cross-encoder/
DATA_DIR = BASE_DIR / "data"              # cross-encoder/data/

EVAL_DIR = ROOT / "outputs" / "eval"     # notebooks/outputs/eval/
TMP_DIR  = ROOT / "outputs" / "tmp"      # notebooks/outputs/tmp/
MDL_DIR  = BASE_DIR / "outputs" / "models"  # cross-encoder/outputs/models/ (share)

EVAL_QA_FILE = EVAL_DIR / "eval_qa.jsonl"

TRAIN_FILE = DATA_DIR / "train.jsonl"
DEV_FILE   = DATA_DIR / "dev.jsonl"
TRAIN_NEG  = DATA_DIR / "train_with_neg.jsonl"


# ── Eval QA loader (tự tạo từ dev.jsonl nếu chưa có) ──
def get_eval_qa():
    """
    Trả về list {id, query, expected_citations}.
    Nếu eval_qa.jsonl chưa có → tạo từ dev.jsonl (label=1).
    """
    if EVAL_QA_FILE.exists():
        return load_jsonl(EVAL_QA_FILE)

    print(f"  eval_qa.jsonl not found — building from {DEV_FILE.name} ...")
    EVAL_QA_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(load_jsonl(DEV_FILE)):
        if r.get("label", 1) != 1:
            continue
        meta = r.get("meta", {})
        rows.append({
            "id":    f"eval_{i:04d}",
            "query": r["query"],
            "expected_citations": [{
                "van_ban":     meta.get("van_ban", ""),
                "chuong":      meta.get("chuong", ""),
                "dieu":        meta.get("dieu", ""),
                "khoan":       meta.get("khoan", ""),
                "diem":        meta.get("diem", ""),
                "chunk_index": meta.get("chunk_index", -1),
            }],
        })
    write_jsonl(EVAL_QA_FILE, rows)
    print(f"  eval_qa.jsonl created: {len(rows)} queries → {EVAL_QA_FILE}")
    return rows



# ── Data I/O ─────────────────────────────────────
def load_jsonl(path, max_rows=None):
    rows, errors = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1
    if errors:
        print(f"  ⚠ {errors} malformed lines in {Path(path).name}")
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else 0.0


# ── Corpus ───────────────────────────────────────
def build_corpus():
    """Gom tất cả unique passages từ train/dev/train_neg."""
    seen = {}
    for path in [TRAIN_FILE, DEV_FILE, TRAIN_NEG]:
        if not Path(path).exists():
            continue
        for r in load_jsonl(path):
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
                }
    corpus = list(seen.values())
    print(f"Corpus: {len(corpus)} passages")
    return corpus


# ── Hit check ────────────────────────────────────
def is_hit(idx, expected_citations, corpus):
    row = corpus[idx]
    for ec in expected_citations:
        ci = ec.get("chunk_index", -2)
        if ci != -1 and row["chunk_index"] == ci:
            return True
        if (row["van_ban"] == ec.get("van_ban", "") and
                row["dieu"]    == ec.get("dieu",    "") and
                row["khoan"]   == ec.get("khoan",   "")):
            return True
    return False


# ── Result entry ─────────────────────────────────
def build_result_entry(item, top_ids, corpus, score_map=None,
                        rank_bi=-1, rank_ce=-1, hits_at=None):
    """Tạo 1 entry JSONL chuẩn format cho mọi version."""
    ec      = item["expected_citations"]
    top5    = []
    mrr     = 0.0
    hits_at = hits_at or {}

    for rank, idx in enumerate(top_ids[:5], 1):
        if idx < 0 or idx >= len(corpus):
            continue
        m   = corpus[idx]
        hit = is_hit(idx, ec, corpus)
        if hit and mrr == 0.0:
            mrr = 1.0 / rank
        top5.append({
            "rank":        rank,
            "hit":         hit,
            "ce_score":    round(score_map.get(idx, 0.0), 4) if score_map else None,
            "van_ban":     m.get("van_ban", ""),
            "chuong":      m.get("chuong", ""),
            "dieu":        m.get("dieu", ""),
            "khoan":       m.get("khoan", ""),
            "diem":        m.get("diem", ""),
            "chunk_index": m.get("chunk_index", -1),
            "passage":     m.get("passage", ""),
        })

    return {
        "id":                 item.get("id", ""),
        "query":              item["query"],
        "expected_citations": ec,
        "rank_bi":            rank_bi,
        "rank_ce":            rank_ce,
        "hit@1":              hits_at.get(1, 0),
        "hit@3":              hits_at.get(3, 0),
        "hit@5":              hits_at.get(5, 0),
        "mrr":                round(mrr, 4),
        "top5_reranked":      top5,
    }


# ── Metrics ──────────────────────────────────────
def compute_metrics(per_query):
    m = {"Recall@1": [], "Recall@3": [], "Recall@5": [], "MRR@10": []}
    for r in per_query:
        m["Recall@1"].append(r.get("hit@1", 0))
        m["Recall@3"].append(r.get("hit@3", 0))
        m["Recall@5"].append(r.get("hit@5", 0))
        m["MRR@10"].append(r.get("mrr", 0.0))
    return {k: avg(v) for k, v in m.items()}


def print_metrics(metrics, version_label=""):
    print(f"\n── {version_label} Results ──")
    print(f"  {'Metric':<12} {'Value':>10}")
    print("  " + "-" * 24)
    for metric, val in metrics.items():
        print(f"  {metric:<12} {val:>10.4f}")


def save_csv(metrics, out_path, extra_cols=None):
    rows = []
    for metric, val in metrics.items():
        row = {"metric": metric, "value": val}
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved → {out_path} ✓")
