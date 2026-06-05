#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test retrieval of newly ingested document chunks.
Uses production components to query and check if the given doc_id chunks are retrieved.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# Set standard streams to UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Setup path routing
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "ChatBot"))
sys.path.append(str(ROOT_DIR / "ChatBot" / "generation"))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("retrieval_test")

STAGING_REPORTS = ROOT_DIR / "vector_data" / "staging" / "reports"

def main():
    parser = argparse.ArgumentParser(description="Test retrieval of an ingested document.")
    parser.add_argument("--doc-id", required=True, help="Document ID to verify")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of retrieved chunks to return")
    args = parser.parse_args()

    t0 = time.time()
    logger.info(f"Initializing LegalRetriever on CPU for testing...")
    
    # Import retriever components
    try:
        from generation.run_generation import LegalRetriever
    except ImportError as e:
        logger.error(f"Failed to import LegalRetriever: {e}")
        sys.exit(1)

    retriever = LegalRetriever(device="cpu")
    
    # Verify metadata contains doc_id before querying
    logger.info(f"Checking if doc_id '{args.doc_id}' exists in production metadata...")
    doc_chunks_count = 0
    if hasattr(retriever, 'faiss_mapping'):
        for entry in retriever.faiss_mapping:
            meta = entry.get("metadata", {})
            if meta.get("doc_id") == args.doc_id:
                doc_chunks_count += 1
    
    logger.info(f"Found {doc_chunks_count} chunks matching doc_id '{args.doc_id}' in metadata.")
    if doc_chunks_count == 0:
        logger.warning(f"doc_id '{args.doc_id}' not found in production metadata.json.")

    logger.info(f"Executing query: '{args.query}' (top-k={args.top_k})...")
    # Perform retrieve and rerank using production hybrid setup
    candidates = retriever.retrieve_and_rerank(args.query, top_k_rerank=args.top_k)
    
    logger.info(f"Retrieved {len(candidates)} candidates.")
    
    # Print results table
    print("\n" + "="*80)
    print(f"RETRIEVAL RESULTS FOR: '{args.query}'")
    print("="*80)
    
    top_results_data = []
    found_target_doc = False
    
    for idx, cand in enumerate(candidates):
        rank = idx + 1
        
        # Extract metadata from production mapping using the candidate's chunk_id
        meta = {}
        if hasattr(retriever, 'faiss_mapping') and 0 <= cand.chunk_id < len(retriever.faiss_mapping):
            meta = retriever.faiss_mapping[cand.chunk_id].get("metadata", {})
            
        cand_doc_id = meta.get("doc_id", "N/A")
        cand_file_name = meta.get("file_name", "N/A")
        cand_so_hieu = meta.get("so_ky_hieu", "N/A")
        cand_dieu = meta.get("dieu", cand.dieu or "N/A")
        cand_khoan = meta.get("khoan", cand.khoan or "N/A")
        cand_diem = meta.get("diem", cand.diem or "N/A")
        
        if cand_doc_id == args.doc_id:
            found_target_doc = True
            marker = "🎯 [NEW DOC]"
        else:
            marker = ""
            
        print(f"Rank {rank} | Score: {cand.score_rerank:.4f} {marker}")
        print(f"  - Doc ID:      {cand_doc_id}")
        print(f"  - File Name:   {cand_file_name}")
        print(f"  - So Hieu:     {cand_so_hieu}")
        print(f"  - Structure:   Dieu {cand_dieu} | Khoan {cand_khoan} | Diem {cand_diem}")
        print(f"  - Preview:     {cand.text[:300]}...")
        print("-" * 80)
        
        top_results_data.append({
            "rank": rank,
            "score": cand.score_rerank,
            "doc_id": cand_doc_id,
            "chunk_id": cand.chunk_id,
            "file_name": cand_file_name,
            "so_ky_hieu": cand_so_hieu,
            "dieu": cand_dieu,
            "khoan": cand_khoan,
            "diem": cand_diem,
            "text_preview": cand.text[:300]
        })

    # Summary analysis
    duration = time.time() - t0
    logger.info(f"Retrieval analysis completed in {duration:.2f}s.")
    if found_target_doc:
        logger.info(f"SUCCESS: Found chunk of newly ingested doc_id '{args.doc_id}' in top-k results.")
    else:
        logger.warning(f"NOTICE: Newly ingested doc_id '{args.doc_id}' chunks did not appear in top-k results for this query.")

    # Write report
    report_path = STAGING_REPORTS / f"{args.doc_id}_retrieval_test.json"
    STAGING_REPORTS.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "doc_id": args.doc_id,
        "query": args.query,
        "total_chunks_in_metadata": doc_chunks_count,
        "found_in_top_k": found_target_doc,
        "duration_seconds": duration,
        "retrieved_results": top_results_data
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Retrieval test report written to: {report_path}")

if __name__ == "__main__":
    main()
