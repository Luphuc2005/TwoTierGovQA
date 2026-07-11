#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orchestrator script to ingest a new PDF into the Legal RAG production vector store.
Supports dry-run and commit modes.
"""

import os
import sys
import json
import argparse
import hashlib
import shutil
import time
import re
from pathlib import Path
import traceback
import numpy as np

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
sys.path.append(str(ROOT_DIR / "ChatBot" / "xldl"))
sys.path.append(str(ROOT_DIR / "ChatBot" / "generation"))

# Log setup
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ingest_pipeline")

# Constants
PRODUCTION_DIR = ROOT_DIR / "vector_data" / "production"
FAISS_PATH = PRODUCTION_DIR / "index.faiss"
METADATA_PATH = PRODUCTION_DIR / "metadata.json"
BM25_PATH = PRODUCTION_DIR / "bm25.pkl"

# Staging directories
STAGING_DIR = ROOT_DIR / "vector_data" / "staging"
UPLOADS_DIR = ROOT_DIR / "data" / "uploads"
BACKUPS_DIR = ROOT_DIR / "vector_data" / "backups"

STAGING_OCR = STAGING_DIR / "ocr"
STAGING_FULL_TEXT = STAGING_DIR / "full_text"
STAGING_PARSED = STAGING_DIR / "parsed"
STAGING_CHUNKS = STAGING_DIR / "chunks"
STAGING_EMBEDDINGS = STAGING_DIR / "embeddings"
STAGING_REPORTS = STAGING_DIR / "reports"

# Environment / Paths Fallback
HF_HOME = os.environ.get('HF_HOME', 'D:/huggingface_cache')
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', HF_HOME)

BI_MODEL_PATH = ROOT_DIR / "ChatBot" / "cross-encoder" / "outputs" / "models" / "bi_bge_m3_ft"

# Target dimensions
EMBEDDING_DIM = 1024

def init_directories():
    """Create staging, backup, and upload directories if they don't exist."""
    for folder in [
        UPLOADS_DIR, STAGING_DIR, STAGING_OCR, STAGING_FULL_TEXT,
        STAGING_PARSED, STAGING_CHUNKS, STAGING_EMBEDDINGS,
        STAGING_REPORTS, BACKUPS_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)

def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def extract_digital_text(pdf_path: Path) -> dict:
    """
    Attempt to extract digital text from PDF if available (non-scan).
    Returns mock OCR results dict or None if text is too sparse.
    """
    logger.info("Checking for digital text inside the PDF...")
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not available, skipping digital text check")
        return None

    results = {}
    total_chars = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_num = idx + 1
                text = page.extract_text()
                if text:
                    text_str = text.strip()
                    total_chars += len(text_str)
                    results[f"page_{page_num}"] = {
                        "page_number": page_num,
                        "text": text_str,
                        "boxes": []
                    }
    except Exception as e:
        logger.warning(f"Error checking digital text: {e}")
        return None

    # Threshold: If less than 100 characters total, treat as scanned PDF
    if total_chars < 100:
        logger.info("PDF contains very little digital text. Will run OCR pipeline.")
        return None

    logger.info(f"Digital text successfully extracted ({total_chars} chars across {len(results)} pages). Skipping OCR.")
    return results

def get_vietnamese_tokenizer():
    """Get the local tokenizer function matching the production pipeline."""
    # Matches LegalRetriever._tokenize_vi
    def _tokenize_vi(text: str) -> list:
        text = text.lower()
        tokens = re.findall(
            r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9]+',
            text,
        )
        return tokens if tokens else text.split()
    return _tokenize_vi

def rebuild_bm25_from_metadata(metadata: list, output_path: Path):
    """Rebuild BM25 index from corpus metadata."""
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

class IngestPipeline:
    def __init__(self, pdf_path: str, dry_run: bool = True):
        self.raw_pdf_path = Path(pdf_path)
        self.dry_run = dry_run
        self.warnings = []
        self.errors = []
        self.created_files = []
        
        # Pipeline outputs
        self.doc_id = ""
        self.file_sha256 = ""
        self.staged_pdf_path = None
        self.ocr_json_path = None
        self.full_text_path = None
        self.parsed_json_path = None
        self.chunks_json_path = None
        self.embeddings_npy_path = None
        self.report_path = None
        
        self.num_chunks = 0
        self.text_length = 0
        self.embedding_dim = 0
        
        # Production status
        self.faiss_current_ntotal = 0
        self.metadata_current_count = 0
        self.bm25_current_count = 0

    def load_production_stats(self):
        """Load production vector store statistics if files exist."""
        import faiss
        
        if FAISS_PATH.exists():
            try:
                idx = faiss.read_index(str(FAISS_PATH))
                self.faiss_current_ntotal = idx.ntotal
            except Exception as e:
                self.warnings.append(f"Could not read production FAISS index size: {e}")
        
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.metadata_current_count = len(meta)
            except Exception as e:
                self.warnings.append(f"Could not read production metadata size: {e}")

        if BM25_PATH.exists():
            try:
                import pickle
                with open(BM25_PATH, "rb") as f:
                    bm25_data = pickle.load(f)
                    self.bm25_current_count = len(bm25_data.get("corpus", []))
            except Exception as e:
                self.warnings.append(f"Could not read production BM25 corpus size: {e}")

    def run_ingest(self) -> bool:
        """Execute the ingestion pipeline."""
        t0 = time.time()
        logger.info(f"Starting Ingest Pipeline (mode: {'dry-run' if self.dry_run else 'commit'})")
        
        init_directories()
        self.load_production_stats()
        
        try:
            # 1. Validate PDF
            if not self.step_validate_pdf():
                self.write_report("failed")
                return False
                
            # 2. Extract text (Digital first, fall back to OCR)
            if not self.step_ocr():
                self.write_report("failed")
                return False
                
            # 3. Clean and merge
            if not self.step_merge():
                self.write_report("failed")
                return False
                
            # 4. Parse legal structure
            if not self.step_parse():
                self.write_report("failed")
                return False
                
            # 5. Segment into chunks
            if not self.step_chunk():
                self.write_report("failed")
                return False
                
            # 6. Generate embeddings
            if not self.step_embed():
                self.write_report("failed")
                return False
                
            # 7. Merge to production (if commit)
            if not self.dry_run:
                if not self.step_commit():
                    self.write_report("failed")
                    return False
            
            logger.info(f"Pipeline completed successfully in {time.time() - t0:.2f}s!")
            self.write_report("success")
            return True
            
        except Exception as e:
            err_msg = f"Pipeline execution failed: {e}\n{traceback.format_exc()}"
            logger.error(err_msg)
            self.errors.append(err_msg)
            self.write_report("failed")
            return False

    def step_validate_pdf(self) -> bool:
        """Validate input file attributes and duplicate hash."""
        logger.info("--- Step 1: Validating PDF Input ---")
        if not self.raw_pdf_path.exists():
            self.errors.append(f"Input file does not exist: {self.raw_pdf_path}")
            return False
            
        if self.raw_pdf_path.suffix.lower() != ".pdf":
            self.errors.append(f"Input file is not a PDF: {self.raw_pdf_path}")
            return False
            
        file_size = self.raw_pdf_path.stat().st_size
        if file_size == 0:
            self.errors.append("PDF file is empty (0 bytes)")
            return False
            
        if file_size > 100 * 1024 * 1024:
            self.errors.append("PDF file exceeds maximum size of 100MB")
            return False

        # Compute file hash
        self.file_sha256 = compute_sha256(self.raw_pdf_path)
        self.doc_id = f"doc_{self.file_sha256[:16]}"
        logger.info(f"PDF SHA-256: {self.file_sha256}")
        logger.info(f"Generated Doc ID: {self.doc_id}")

        # Staged PDF Copy
        self.staged_pdf_path = UPLOADS_DIR / f"{self.doc_id}.pdf"
        if not self.staged_pdf_path.exists():
            shutil.copy(self.raw_pdf_path, self.staged_pdf_path)
            self.created_files.append(str(self.staged_pdf_path))
            logger.info(f"Staged PDF copied to: {self.staged_pdf_path}")

        # Check duplication in production metadata
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    metadata_list = json.load(f)
                    duplicate_count = sum(
                        1
                        for item in metadata_list
                        if item.get("metadata", {}).get("file_sha256") == self.file_sha256
                    )
                    if duplicate_count:
                        duplicate_msg = (
                            f"Document with SHA-256 {self.file_sha256} already exists "
                            f"in production metadata ({duplicate_count} chunks)."
                        )
                        if self.dry_run:
                            self.warnings.append(f"{duplicate_msg} Continuing dry-run.")
                        else:
                            self.errors.append(f"Duplicate document error: {duplicate_msg}")
                            return False
            except Exception as e:
                self.warnings.append(f"Error checking duplicate SHA-256: {e}")
        else:
            self.warnings.append("Production metadata.json not found; skipping duplication check.")

        logger.info("PDF validation passed.")
        return True

    def step_ocr(self) -> bool:
        """Run OCR pipeline (or extract digital text fallback)."""
        logger.info("--- Step 2: Extraction (OCR / Digital Text) ---")
        self.ocr_json_path = STAGING_OCR / f"{self.doc_id}_protonx_ocr.json"
        
        # Check if already processed in staging
        if self.ocr_json_path.exists():
            logger.info(f"Reusing existing OCR results from staging: {self.ocr_json_path}")
            try:
                with open(self.ocr_json_path, "r", encoding="utf-8") as f:
                    ocr_data = json.load(f)
                if ocr_data and len(ocr_data) > 0:
                    return True
            except Exception as e:
                logger.warning(f"Existing staged OCR file is invalid: {e}. Re-processing...")

        # Try digital text fallback first
        digital_results = extract_digital_text(self.staged_pdf_path)
        if digital_results:
            with open(self.ocr_json_path, "w", encoding="utf-8") as f:
                json.dump(digital_results, f, ensure_ascii=False, indent=2)
            self.created_files.append(str(self.ocr_json_path))
            return True

        # Scanned PDF: Run OCR
        logger.info("Running OCR on Scanned PDF...")
        try:
            from OCR_paddle_protonX import process_pdf as process_pdf_ocr
        except ImportError as e:
            self.errors.append(f"Failed to import OCR dependencies: {e}. Missing paddleocr, vietocr, pdf2image, or poppler. Cannot parse scanned PDF.")
            return False

        try:
            # Check poppler
            poppler_bin = os.environ.get('POPPLER_PATH', r"D:\Release-25.12.0-0\poppler-25.12.0\Library\bin")
            if not Path(poppler_bin).exists():
                self.warnings.append(f"Poppler path '{poppler_bin}' does not exist. PDF-to-image conversion might fail.")

            ocr_results = process_pdf_ocr(str(self.staged_pdf_path), str(self.ocr_json_path))
            self.created_files.append(str(self.ocr_json_path))

            # Validate OCR results
            if not ocr_results:
                self.errors.append("OCR pipeline returned empty results.")
                return False

            total_text = ""
            for key, val in ocr_results.items():
                total_text += val.get("text", "")
            
            if len(total_text.strip()) < 100:
                self.errors.append("OCR output text is too short (< 100 characters). Recognition may have failed.")
                return False

            logger.info(f"OCR completed. Saved to {self.ocr_json_path}")
            return True

        except Exception as e:
            self.errors.append(f"OCR step failed: {e}\n{traceback.format_exc()}")
            return False

    def step_merge(self) -> bool:
        """Run text merge and clean step."""
        logger.info("--- Step 3: Text Merging & Cleaning ---")
        self.full_text_path = STAGING_FULL_TEXT / f"{self.doc_id}_full.txt"
        
        try:
            with open(self.ocr_json_path, "r", encoding="utf-8") as f:
                ocr_results = json.load(f)

            from rules_base_protonx import merge_ocr_to_text
            full_text = merge_ocr_to_text(ocr_results)
            self.text_length = len(full_text)
            logger.info(f"Merged full text length: {self.text_length} chars")

            # Validate merged text requirements
            if self.text_length < 500:
                self.errors.append(f"Full text is too short ({self.text_length} chars). Minimum requirement is 500 chars.")
                return False

            # Check keywords (support accented, unaccented, and semi-accented variants due to OCR limitations)
            keywords = [
                "căn cứ", "điều", "khoản", "nghị định", "quyết định", "thông tư", "ủy ban nhân dân", "chính phủ",
                "can cu", "cän cu", "dieu", "diéu", "diều", "khoan", "nghi dinh", "quyet dinh", "thong tu", "uy ban nhan dan", "chinh phu"
            ]
            found_kws = [kw for kw in keywords if kw in full_text.lower()]
            logger.info(f"Found legal keywords: {found_kws}")
            if len(found_kws) < 2:
                self.errors.append(f"Failed keyword validation. Required at least 2 legal terms, found: {found_kws}")
                return False

            # Check page marker
            if "[PAGE_" not in full_text:
                self.warnings.append("No page markers '[PAGE_N]' found in merged full text. Citation formatting may be degraded.")

            # Save full text
            with open(self.full_text_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            self.created_files.append(str(self.full_text_path))
            
            logger.info(f"Merged and cleaned text saved to: {self.full_text_path}")
            return True

        except Exception as e:
            self.errors.append(f"Text merging step failed: {e}")
            return False

    def step_parse(self) -> bool:
        """Parse structured articles from merged text."""
        logger.info("--- Step 4: Legal Document Structure Parsing ---")
        self.parsed_json_path = STAGING_PARSED / f"{self.doc_id}_parsed.json"
        
        try:
            with open(self.full_text_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            from legal_parser import LawParser
            parser = LawParser()
            parsed_result = parser.parse(full_text)

            # Check parsed structures
            has_chapters = len(parsed_result.get("chuong", [])) > 0
            has_title = bool(parsed_result.get("ten_van_ban") and parsed_result["ten_van_ban"] != "Không xác định")

            quality = "high"
            if not has_title:
                self.warnings.append("Document title ('ten_van_ban') was not parsed or set to 'Không xác định'.")
                quality = "low"
            if not has_chapters:
                # Some short decrees or decisions do not have chapters; warnings instead of failure
                self.warnings.append("No chapters parsed. Checking for flat articles structure...")
                
            parsed_result["metadata_quality"] = quality
            parsed_result["source_file"] = str(self.staged_pdf_path)

            with open(self.parsed_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_result, f, ensure_ascii=False, indent=2)
            self.created_files.append(str(self.parsed_json_path))

            logger.info(f"Structure parsed. Quality: {quality}. Saved to: {self.parsed_json_path}")
            return True

        except Exception as e:
            self.errors.append(f"Parsing step failed: {e}")
            return False

    def step_chunk(self) -> bool:
        """Create granular chunks from parsed structure."""
        logger.info("--- Step 5: Document Chunking ---")
        self.chunks_json_path = STAGING_CHUNKS / f"{self.doc_id}_chunks.json"
        
        try:
            # We MUST run chunks.py build_chunks_from_file by saving *_final.json to staging first,
            # then calling the imported function build_chunks_from_file.
            final_json_path = STAGING_PARSED / f"{self.doc_id}_final.json"
            shutil.copy(self.parsed_json_path, final_json_path)
            self.created_files.append(str(final_json_path))

            from chunks import build_chunks_from_file
            chunks = build_chunks_from_file(str(final_json_path))

            # Validate chunks
            if not chunks:
                self.errors.append("No chunks were generated by chunker.")
                return False

            logger.info(f"Generated {len(chunks)} raw chunks.")

            # Load parsed final metadata for backward-compatible properties enrichment
            with open(final_json_path, "r", encoding="utf-8") as f:
                parsed_final = json.load(f)
            
            # Extract document metadata from full text using regex fallback
            doc_metadata = {"so_hieu": "", "loai_van_ban": "", "co_quan_ban_hanh": "", "ngay_ban_hanh": ""}
            try:
                with open(self.full_text_path, "r", encoding="utf-8") as f:
                    full_text_content = f.read()
                
                # 1. Extract Số ký hiệu
                so_hieu_match = re.search(r'Số:\s*([^\s\n\r]+)', full_text_content, re.IGNORECASE)
                if so_hieu_match:
                    doc_metadata["so_hieu"] = so_hieu_match.group(1).strip()
                
                # 2. Extract Ngày ban hành
                date_match = re.search(r'ngày\s+(\d+)\s+tháng\s+(\d+)\s+năm\s+(\d+)', full_text_content, re.IGNORECASE)
                if date_match:
                    doc_metadata["ngay_ban_hanh"] = f"{date_match.group(1)}/{date_match.group(2)}/{date_match.group(3)}"
                
                # 3. Extract Loại văn bản
                doc_types = ["NGHỊ ĐỊNH", "QUYẾT ĐỊNH", "THÔNG TƯ", "LUẬT", "NGHỊ QUYẾT", "CHỈ THỊ", "PHÁP LỆNH"]
                for doc_type in doc_types:
                    if re.search(r'\b' + re.escape(doc_type) + r'\b', full_text_content, re.IGNORECASE):
                        doc_metadata["loai_van_ban"] = doc_type
                        break
                
                # 4. Extract Cơ quan ban hành
                lines = full_text_content.splitlines()
                for idx, line in enumerate(lines[:30]):
                    if "Độc lập" in line or "ĐỘC LẬP" in line:
                        for offset in [1, 2, 3]:
                            if idx - offset >= 0:
                                prev_line = lines[idx - offset].strip()
                                if prev_line and prev_line not in ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "CONG HOA XA HOI", ""]:
                                    if not any(k in prev_line for k in ["NGƯỜI KÝ", "Email", "Thời gian", "Cơ quan:"]):
                                        doc_metadata["co_quan_ban_hanh"] = prev_line
                                        break
                        break
                logger.info(f"Extracted metadata from text: {doc_metadata}")
            except Exception as meta_err:
                logger.warning(f"Failed to extract document metadata from text: {meta_err}")

            # Save metadata back to parsed_final
            parsed_final["metadata"] = {
                "so_hieu": parsed_final.get("metadata", {}).get("so_hieu", "") or doc_metadata["so_hieu"],
                "loai_van_ban": parsed_final.get("metadata", {}).get("loai_van_ban", "") or doc_metadata["loai_van_ban"],
                "co_quan_ban_hanh": parsed_final.get("metadata", {}).get("co_quan_ban_hanh", "") or doc_metadata["co_quan_ban_hanh"],
                "ngay_ban_hanh": parsed_final.get("metadata", {}).get("ngay_ban_hanh", "") or doc_metadata["ngay_ban_hanh"]
            }
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_final, f, ensure_ascii=False, indent=2)
            with open(self.parsed_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_final, f, ensure_ascii=False, indent=2)

            metadata_quality = parsed_final.get("metadata_quality", "high")
            ten_van_ban = parsed_final.get("ten_van_ban", "Không xác định")

            # Enrich and validate chunks
            enriched_chunks = []
            seen_texts = set()

            for idx, chk in enumerate(chunks):
                txt = chk.get("text", "").strip()
                if not txt:
                    continue
                if len(txt) < 10:
                    self.warnings.append(f"Chunk {idx} is unusually short: '{txt}'")
                
                # Check duplicate text in same document
                text_hash = hashlib.md5(txt.encode('utf-8')).hexdigest()
                if text_hash in seen_texts:
                    self.warnings.append(f"Skipping duplicate chunk text at index {idx}")
                    continue
                seen_texts.add(text_hash)

                orig_meta = chk.get("metadata", {})
                
                # Dynamic page start/end from chunk metadata
                page_start = orig_meta.get("page_number")
                page_end = page_start

                # Match backward-compatible metadata schema plus requested extra fields
                enriched_meta = {
                    "chunk_index": idx,
                    "chuong": orig_meta.get("chuong"),
                    "diem": orig_meta.get("diem"),
                    "dieu": orig_meta.get("dieu"),
                    "khoan": orig_meta.get("khoan"),
                    "van_ban": ten_van_ban,
                    
                    # Extra fields requested
                    "doc_id": self.doc_id,
                    "file_sha256": self.file_sha256,
                    "file_name": self.raw_pdf_path.name,
                    "source_pdf": str(self.staged_pdf_path),
                    "so_ky_hieu": parsed_final.get("metadata", {}).get("so_hieu", ""),
                    "loai_van_ban": parsed_final.get("metadata", {}).get("loai_van_ban", ""),
                    "co_quan_ban_hanh": parsed_final.get("metadata", {}).get("co_quan_ban_hanh", ""),
                    "ngay_ban_hanh": parsed_final.get("metadata", {}).get("ngay_ban_hanh", ""),
                    "page_start": page_start,
                    "page_end": page_end,
                    "metadata_quality": metadata_quality
                }

                enriched_chunks.append({
                    "text": txt,
                    "metadata": enriched_meta
                })

            self.num_chunks = len(enriched_chunks)
            if self.num_chunks == 0:
                self.errors.append("No valid unique chunks left after cleaning.")
                return False

            with open(self.chunks_json_path, "w", encoding="utf-8") as f:
                json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)
            self.created_files.append(str(self.chunks_json_path))

            logger.info(f"Chunking validation completed. {self.num_chunks} chunks written to: {self.chunks_json_path}")
            return True

        except Exception as e:
            self.errors.append(f"Chunking step failed: {e}\n{traceback.format_exc()}")
            return False

    def step_embed(self) -> bool:
        """Load local embedding model, encode chunks, and validate dimensions."""
        logger.info("--- Step 6: Text Embedding Generation ---")
        self.embeddings_npy_path = STAGING_EMBEDDINGS / f"{self.doc_id}_embeddings.npy"
        
        try:
            with open(self.chunks_json_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            texts = [c["text"] for c in chunks]

            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading local embedding model from: {BI_MODEL_PATH}")
            if not BI_MODEL_PATH.exists():
                self.errors.append(f"Embedding model directory not found: {BI_MODEL_PATH}")
                return False

            # Load model
            model = SentenceTransformer(str(BI_MODEL_PATH))
            
            # Encode with L2 normalize setting as production requires
            logger.info(f"Encoding {len(texts)} chunks...")
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            embeddings = np.array(embeddings).astype("float32")

            # Validate dimensions
            self.embedding_dim = embeddings.shape[1]
            logger.info(f"Embeddings shape: {embeddings.shape}")
            if self.embedding_dim != EMBEDDING_DIM:
                self.errors.append(f"Dimension mismatch! Expected {EMBEDDING_DIM}, got {self.embedding_dim}")
                return False

            if embeddings.shape[0] != self.num_chunks:
                self.errors.append(f"Count mismatch! Chunks count={self.num_chunks}, embeddings count={embeddings.shape[0]}")
                return False

            # Save staged embeddings
            np.save(str(self.embeddings_npy_path), embeddings)
            self.created_files.append(str(self.embeddings_npy_path))

            logger.info(f"Embeddings saved to: {self.embeddings_npy_path}")
            return True

        except Exception as e:
            self.errors.append(f"Embedding step failed: {e}\n{traceback.format_exc()}")
            return False

    def step_commit(self) -> bool:
        """Safely backup and merge staged chunks and vectors to production."""
        logger.info("--- Step 7: Committing to Production ---")
        import faiss
        
        backup_dir = None
        temp_faiss = None
        temp_meta = None
        temp_bm25 = None
        
        try:
            # 1. Load production
            logger.info("Verifying production files existence...")
            if not FAISS_PATH.exists() or not METADATA_PATH.exists() or not BM25_PATH.exists():
                # If production store is empty or missing, raise error
                self.errors.append("Production vector store files missing. Run initial build index first.")
                return False

            index = faiss.read_index(str(FAISS_PATH))
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # 2. Check initial consistency
            logger.info(f"Production initial count: FAISS={index.ntotal}, Metadata={len(metadata)}")
            if index.ntotal != len(metadata):
                self.errors.append(f"Consistency error before merge: FAISS count ({index.ntotal}) != Metadata count ({len(metadata)})")
                return False

            # 3. Create Backup
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = BACKUPS_DIR / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating production backup in {backup_dir}...")
            
            shutil.copy(FAISS_PATH, backup_dir / "index.faiss")
            shutil.copy(METADATA_PATH, backup_dir / "metadata.json")
            shutil.copy(BM25_PATH, backup_dir / "bm25.pkl")
            logger.info("Backup created successfully.")

            # 4. Load staged new embeddings
            new_embeddings = np.load(str(self.embeddings_npy_path)).astype("float32")
            if index.d != new_embeddings.shape[1]:
                self.errors.append(f"Dimension mismatch! Production FAISS dim={index.d}, new embeddings dim={new_embeddings.shape[1]}")
                return False

            # 5. Merge new vectors to FAISS
            logger.info(f"Adding {new_embeddings.shape[0]} vectors to FAISS index...")
            index.add(new_embeddings)

            # 6. Load staged chunks and append to metadata
            with open(self.chunks_json_path, "r", encoding="utf-8") as f:
                new_chunks = json.load(f)
            
            updated_metadata = metadata + new_chunks
            logger.info(f"New combined metadata size: {len(updated_metadata)}")

            # 7. Write FAISS and Metadata safely using .tmp files
            temp_faiss = FAISS_PATH.with_suffix(".tmp")
            temp_meta = METADATA_PATH.with_suffix(".tmp")
            temp_bm25 = BM25_PATH.with_suffix(".tmp")

            faiss.write_index(index, str(temp_faiss))
            with open(temp_meta, "w", encoding="utf-8") as f:
                json.dump(updated_metadata, f, ensure_ascii=False, indent=2)

            # 8. Rebuild BM25 safely
            rebuild_bm25_from_metadata(updated_metadata, temp_bm25)

            # 9. Verify consistency of temp files before replacing
            temp_idx = faiss.read_index(str(temp_faiss))
            with open(temp_meta, "r", encoding="utf-8") as f:
                temp_metadata_list = json.load(f)
            
            import pickle
            with open(temp_bm25, "rb") as f:
                temp_bm25_data = pickle.load(f)
                temp_bm25_corpus_size = len(temp_bm25_data.get("corpus", []))

            logger.info(f"Validating temp files: FAISS={temp_idx.ntotal}, Metadata={len(temp_metadata_list)}, BM25={temp_bm25_corpus_size}")
            if temp_idx.ntotal != len(temp_metadata_list) or temp_idx.ntotal != temp_bm25_corpus_size:
                raise Exception("Consistency verification failed for temporary updated store files.")

            # 10. Atomically replace production files
            if FAISS_PATH.exists(): FAISS_PATH.unlink()
            temp_faiss.rename(FAISS_PATH)

            if METADATA_PATH.exists(): METADATA_PATH.unlink()
            temp_meta.rename(METADATA_PATH)

            if BM25_PATH.exists(): BM25_PATH.unlink()
            temp_bm25.rename(BM25_PATH)

            logger.info("Production vector store and index committed and updated successfully.")
            return True

        except Exception as e:
            err_msg = f"Commit step failed: {e}. Starting rollback from backup..."
            logger.error(err_msg)
            self.errors.append(err_msg)
            
            # Clean temp files if left over
            for tmp in [temp_faiss, temp_meta, temp_bm25]:
                if tmp and tmp.exists():
                    try: tmp.unlink()
                    except: pass
            
            # Rollback
            if backup_dir and backup_dir.exists():
                try:
                    logger.info("Rolling back production files from backup...")
                    shutil.copy(backup_dir / "index.faiss", FAISS_PATH)
                    shutil.copy(backup_dir / "metadata.json", METADATA_PATH)
                    shutil.copy(backup_dir / "bm25.pkl", BM25_PATH)
                    logger.info("Rollback complete.")
                except Exception as rollback_err:
                    logger.critical(f"FATAL: Rollback failed! Production may be in an inconsistent state: {rollback_err}")
                    self.errors.append(f"Rollback failed: {rollback_err}")
            
            return False

    def write_report(self, status: str):
        """Generate final json report for the run."""
        self.report_path = STAGING_REPORTS / f"{self.doc_id}_report.json"
        
        report_data = {
            "status": status,
            "mode": "dry-run" if self.dry_run else "commit",
            "doc_id": self.doc_id,
            "file_name": self.raw_pdf_path.name,
            "file_sha256": self.file_sha256,
            "ocr_json_path": str(self.ocr_json_path) if self.ocr_json_path else None,
            "full_text_path": str(self.full_text_path) if self.full_text_path else None,
            "parsed_json_path": str(self.parsed_json_path) if self.parsed_json_path else None,
            "chunks_json_path": str(self.chunks_json_path) if self.chunks_json_path else None,
            "text_length": self.text_length,
            "num_chunks": self.num_chunks,
            "embedding_dim": self.embedding_dim,
            "faiss_current_ntotal": self.faiss_current_ntotal,
            "metadata_current_count": self.metadata_current_count,
            "bm25_current_count": self.bm25_current_count,
            "warnings": self.warnings,
            "errors": self.errors,
            "created_files": self.created_files
        }
        
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Run report written to: {self.report_path}")
            
            # CLI output matching Acceptance Criteria
            print("\n" + "="*50)
            print(f"INGESTION REPORT: {status.upper()}")
            print("="*50)
            print(f"Mode:          {'dry-run' if self.dry_run else 'commit'}")
            print(f"Document ID:   {self.doc_id}")
            print(f"Total Chunks:  {self.num_chunks}")
            print(f"Embedding Dim: {self.embedding_dim}")
            if not self.dry_run and status == "success":
                print(f"FAISS Total:   {self.faiss_current_ntotal + self.num_chunks}")
            else:
                print(f"FAISS Total:   {self.faiss_current_ntotal} (no change)")
            print(f"Warnings:      {len(self.warnings)}")
            print(f"Errors:        {len(self.errors)}")
            print("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to write report file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Ingest a new legal PDF into the production vector store.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Validate and process files without writing to production store")
    group.add_argument("--commit", action="store_true", help="Commit changes directly to the production store with safety backups")
    
    args = parser.parse_args()
    
    pipeline = IngestPipeline(args.pdf, dry_run=args.dry_run)
    success = pipeline.run_ingest()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
