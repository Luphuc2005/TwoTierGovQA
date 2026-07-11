"""
Repair OCR metadata for Nghi quyet 202/2025/QH15, Article 1 Clause 14.

The production chunk lost "Binh Thuan va tinh Lam Dong" during OCR, so queries
mentioning Binh Thuan cannot match the correct merger row by BM25/entity boost.
This script backs up metadata.json and bm25.pkl, patches the chunk text, then
rebuilds BM25 from the updated metadata.
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path

from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = ROOT / "vector_data" / "production"
BACKUP_ROOT = ROOT / "vector_data" / "backups"
METADATA_PATH = PRODUCTION_DIR / "metadata.json"
BM25_PATH = PRODUCTION_DIR / "bm25.pkl"

BAD_PHRASE = (
    "Sắp xếp toàn bộ diện tích tự nhiên, quy mô dân số của tỉnh Đắk Nông, "
    "Sau khi sắp xếp, tỉnh Lâm Đồng"
)
GOOD_PHRASE = (
    "Sắp xếp toàn bộ diện tích tự nhiên, quy mô dân số của tỉnh Đắk Nông, "
    "tỉnh Bình Thuận và tỉnh Lâm Đồng thành tỉnh mới có tên gọi là tỉnh Lâm Đồng. "
    "Sau khi sắp xếp, tỉnh Lâm Đồng"
)


def tokenize_vi(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = re.findall(
        r"[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9]+",
        text,
    )
    return tokens if tokens else text.split()


def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)
    if not BM25_PATH.exists():
        raise FileNotFoundError(BM25_PATH)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"repair_binh_thuan_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(METADATA_PATH, backup_dir / "metadata.json")
    shutil.copy2(BM25_PATH, backup_dir / "bm25.pkl")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    patched = []

    for idx, entry in enumerate(metadata):
        text = entry.get("text", "")
        meta = entry.get("metadata") or {}
        is_target = (
            meta.get("van_ban", "").startswith("Nghị quyết số: 202/2025/QH15")
            and str(meta.get("dieu")) == "1"
            and str(meta.get("khoan")) == "14"
        )
        if is_target and BAD_PHRASE in text:
            entry["text"] = text.replace(BAD_PHRASE, GOOD_PHRASE)
            patched.append(idx)

    if not patched:
        raise RuntimeError("No matching OCR metadata row was patched.")

    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    corpus = [tokenize_vi(entry.get("text", "")) for entry in metadata]
    bm25 = BM25Okapi(corpus)
    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus}, f)

    print(f"patched_indices={patched}")
    print(f"metadata_count={len(metadata)}")
    print(f"backup_dir={backup_dir}")
    print("repair_ok=True")


if __name__ == "__main__":
    main()
