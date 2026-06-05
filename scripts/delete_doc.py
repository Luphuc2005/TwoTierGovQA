#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility script to delete a document from the production vector store (FAISS, metadata, BM25).
Provides backups before making modifications.
"""

import os
import sys
import json
import argparse
import shutil
import time
import re
from pathlib import Path
import numpy as np

# Setup path routing
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "ChatBot"))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("delete_doc")

PRODUCTION_DIR = ROOT_DIR / "vector_data" / "production"
FAISS_PATH = PRODUCTION_DIR / "index.faiss"
METADATA_PATH = PRODUCTION_DIR / "metadata.json"
BM25_PATH = PRODUCTION_DIR / "bm25.pkl"
BACKUPS_DIR = ROOT_DIR / "vector_data" / "backups"

def get_vietnamese_tokenizer():
    def _tokenize_vi(text: str) -> list:
        text = text.lower()
        tokens = re.findall(
            r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9]+',
            text,
        )
        return tokens if tokens else text.split()
    return _tokenize_vi

def rebuild_bm25_from_metadata(metadata: list, output_path: Path):
    from rank_bm25 import BM25Okapi
    import pickle
    
    logger.info(f"Rebuilding BM25 index from {len(metadata)} docs...")
    tokenize = get_vietnamese_tokenizer()
    corpus = [tokenize(entry.get("text", "")) for entry in metadata]
    bm25 = BM25Okapi(corpus)
    
    data = {
        "bm25": bm25,
        "corpus": corpus
    }
    
    tmp_path = output_path.with_suffix(".bm25_tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    if output_path.exists():
        output_path.unlink()
    tmp_path.rename(output_path)
    logger.info(f"BM25 index saved to {output_path}")

def delete_document(doc_id: str) -> bool:
    if not FAISS_PATH.exists() or not METADATA_PATH.exists():
        logger.error("Production vector store files not found.")
        return False

    import faiss

    # 1. Load Metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    # Find matching indices
    match_indices = []
    remaining_metadata = []
    deleted_count = 0

    for idx, item in enumerate(metadata_list):
        item_meta = item.get("metadata", {})
        if item_meta.get("doc_id") == doc_id:
            match_indices.append(idx)
            deleted_count += 1
        else:
            remaining_metadata.append(item)

    if deleted_count == 0:
        logger.warning(f"Document ID '{doc_id}' not found in production metadata.")
        return False

    logger.info(f"Found {deleted_count} chunks matching doc_id '{doc_id}' out of {len(metadata_list)} total chunks.")

    # 2. Create Backup
    timestamp = int(time.time())
    backup_dir = BACKUPS_DIR / f"delete_backup_{doc_id}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating safety backup in: {backup_dir}")
    shutil.copy(FAISS_PATH, backup_dir / "index.faiss")
    shutil.copy(METADATA_PATH, backup_dir / "metadata.json")
    if BM25_PATH.exists():
        shutil.copy(BM25_PATH, backup_dir / "bm25.pkl")

    try:
        # 3. Load FAISS Index and Remove Vectors
        index = faiss.read_index(str(FAISS_PATH))
        logger.info(f"Original FAISS size: {index.ntotal} vectors")

        # Convert to numpy array of IDs to remove
        ids_to_remove = np.array(match_indices, dtype=np.int64)
        index.remove_ids(ids_to_remove)
        logger.info(f"Updated FAISS size: {index.ntotal} vectors")

        # 4. Rebuild BM25
        temp_bm25 = BM25_PATH.with_suffix(".tmp_del")
        rebuild_bm25_from_metadata(remaining_metadata, temp_bm25)

        # 5. Validate Alignment
        if index.ntotal != len(remaining_metadata):
            raise ValueError(f"Alignment mismatch after deletion! FAISS={index.ntotal}, Metadata={len(remaining_metadata)}")

        # 6. Save modified files
        faiss.write_index(index, str(FAISS_PATH))
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(remaining_metadata, f, ensure_ascii=False, indent=2)
        
        if temp_bm25.exists():
            if BM25_PATH.exists():
                BM25_PATH.unlink()
            temp_bm25.rename(BM25_PATH)

        logger.info(f"Successfully deleted document '{doc_id}' from production database.")
        return True

    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        # Restore backups
        logger.info("Restoring files from backup...")
        shutil.copy(backup_dir / "index.faiss", FAISS_PATH)
        shutil.copy(backup_dir / "metadata.json", METADATA_PATH)
        if (backup_dir / "bm25.pkl").exists():
            shutil.copy(backup_dir / "bm25.pkl", BM25_PATH)
        return False

def main():
    parser = argparse.ArgumentParser(description="Delete a document from production vector store.")
    parser.add_argument("--doc-id", required=True, help="Document ID to delete (e.g. doc_db7d4963de2bb467)")
    args = parser.parse_args()

    success = delete_document(args.doc_id)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
