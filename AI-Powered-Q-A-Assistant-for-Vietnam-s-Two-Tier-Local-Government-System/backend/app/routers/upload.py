"""
Upload router — receive PDF files and run the ingestion pipeline with SSE streaming.
"""

import os
import sys
import json
import hashlib
import shutil
import asyncio
import time
import traceback
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from loguru import logger

from app.database import get_db
from app.auth import get_current_user, verify_token
from app.models import User
from app import crud

router = APIRouter(tags=["Upload"])

# ── Paths ─────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent              # routers/
_APP_DIR = _THIS_DIR.parent                               # app/
_BACKEND_DIR = _APP_DIR.parent                            # backend/
_PROJECT_DIR = _BACKEND_DIR.parent                        # AI-Powered-.../
_FOLDER_CHA = _PROJECT_DIR.parent                         # Folder cha/

UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", _FOLDER_CHA / "data" / "uploads"))
STAGING_REPORTS = Path(os.environ.get("STAGING_REPORTS", _FOLDER_CHA / "vector_data" / "staging" / "reports"))
INGEST_SCRIPT = Path(os.environ.get("INGEST_SCRIPT", _FOLDER_CHA / "scripts" / "ingest_new_pdf.py"))
PRODUCTION_METADATA = Path(
    os.environ.get("PRODUCTION_METADATA", _FOLDER_CHA / "vector_data" / "production" / "metadata.json")
)

# Python interpreter — must point to the ChatBot conda env which has
# torch, paddleocr, vietocr, faiss, etc.
PYTHON_EXE = os.environ.get(
    "INGEST_PYTHON_EXE",
    r"C:\Users\ADMIN\anaconda3\envs\ChatBot\python.exe" if os.name == "nt" else sys.executable,
)

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


def _compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _executable_exists(command: str) -> bool:
    """Return True when command is an existing path or resolvable executable."""
    return Path(command).exists() or shutil.which(command) is not None


def _sse(event_name: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


# ── Auth helper for SSE (query param token) ───────────────────

def _get_user_from_token(token: str, db: Session) -> User:
    """Validate JWT from query param (for SSE endpoints where headers aren't available)."""
    payload = verify_token(token)
    user_id = int(payload["sub"])
    user = crud.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ── POST /upload-document ─────────────────────────────────────

@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Accept a PDF file, validate it, save to uploads directory.
    Returns doc_id and file metadata for the streaming step.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tệp PDF.")

    # Validate file size (read content)
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Tệp PDF cần nhỏ hơn 25 MB.")

    # Compute SHA-256 hash
    file_sha256 = hashlib.sha256(content).hexdigest()
    doc_id = f"doc_{file_sha256[:16]}"

    # Save to uploads directory
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / f"{doc_id}.pdf"
    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"📄 PDF uploaded: {file.filename} → {save_path} (doc_id={doc_id})")

    return {
        "doc_id": doc_id,
        "file_name": file.filename,
        "file_size": len(content),
        "file_sha256": file_sha256,
        "saved_path": str(save_path),
    }


# ── GET /upload-document/{doc_id}/stream (SSE) ────────────────

@router.get("/upload-document/{doc_id}/stream")
async def stream_ingestion(
    doc_id: str,
    token: str = Query(..., description="JWT token for authentication"),
    db: Session = Depends(get_db),
):
    """
    Run the ingestion pipeline as a subprocess and stream stdout lines via SSE.
    Uses query-param token because EventSource cannot set Authorization headers.
    """
    # Auth via query param
    _get_user_from_token(token, db)

    # Validate doc_id format
    if not doc_id.startswith("doc_"):
        raise HTTPException(status_code=400, detail="Invalid doc_id format")

    # Verify file exists
    pdf_path = UPLOADS_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {doc_id}")

    async def event_stream():
        """Run ingestion subprocess and yield SSE events."""
        process = None
        started_at = time.time()
        try:
            preflight_errors = []
            if not _executable_exists(PYTHON_EXE):
                preflight_errors.append(
                    f"Ingestion Python not found: {PYTHON_EXE}. Set INGEST_PYTHON_EXE to the ChatBot env python."
                )
            if not INGEST_SCRIPT.exists():
                preflight_errors.append(
                    f"Ingestion script not found: {INGEST_SCRIPT}. Set INGEST_SCRIPT or run backend from the repo root."
                )
            if preflight_errors:
                logger.error("Upload ingestion preflight failed: {}", preflight_errors)
                yield _sse("pipeline_error", {
                    "status": "failed",
                    "doc_id": doc_id,
                    "errors": preflight_errors,
                })
                return

            if PRODUCTION_METADATA.exists():
                try:
                    file_sha256 = _compute_sha256(pdf_path)
                    with open(PRODUCTION_METADATA, "r", encoding="utf-8") as f:
                        metadata_list = json.load(f)

                    existing_chunks = [
                        item for item in metadata_list
                        if item.get("metadata", {}).get("doc_id") == doc_id
                        or item.get("metadata", {}).get("file_sha256") == file_sha256
                    ]
                    if existing_chunks:
                        yield _sse("log", {
                            "line": f"Document already exists in production: {doc_id} ({len(existing_chunks)} chunks).",
                            "step": 5,
                        })
                        yield _sse("complete", {
                            "status": "already_exists",
                            "doc_id": doc_id,
                            "num_chunks": len(existing_chunks),
                            "faiss_total": len(metadata_list),
                            "embedding_dim": 1024,
                        })
                        return
                except Exception as duplicate_check_err:
                    logger.warning(f"Duplicate check skipped for {doc_id}: {duplicate_check_err}")

            cmd = [
                PYTHON_EXE,
                str(INGEST_SCRIPT),
                "--pdf", str(pdf_path),
                "--commit",
            ]

            logger.info(f"🚀 Starting ingestion subprocess for {doc_id}")
            logger.info(f"   CMD: {cmd}")
            logger.info(f"   CWD: {_FOLDER_CHA}")
            logger.info(f"   Python exists: {Path(PYTHON_EXE).exists()}")
            logger.info(f"   Script exists: {INGEST_SCRIPT.exists()}")
            logger.info(f"   PDF exists: {pdf_path.exists()}")
            yield _sse("log", {"line": f"Using Python: {PYTHON_EXE}", "step": 1})
            yield _sse("log", {"line": f"Using script: {INGEST_SCRIPT}", "step": 1})
            yield _sse("log", {"line": f"Working dir: {_FOLDER_CHA}", "step": 1})

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(_FOLDER_CHA),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            # Detect current step from log content
            current_step = 0

            while True:
                raw_line = await asyncio.to_thread(process.stdout.readline)
                if not raw_line:
                    if process.poll() is not None:
                        break
                    await asyncio.sleep(0.05)
                    continue

                line = raw_line.rstrip()
                if not line:
                    continue

                # Detect step transitions from log markers
                new_step = current_step
                if "Step 1:" in line or "Validating PDF" in line:
                    new_step = 1
                elif "Step 2:" in line or "Extraction" in line or "Running OCR" in line:
                    new_step = 2
                elif "Step 3:" in line or "Merging" in line or "Cleaning" in line:
                    new_step = 3
                elif "Step 4:" in line or "Parse" in line or "Legal Document Structure" in line:
                    new_step = 3
                elif "Step 5:" in line or "Document Chunking" in line or "Generated" in line and "chunks" in line:
                    new_step = 4
                elif "Step 6:" in line or "Embedding" in line or "Encoding" in line:
                    new_step = 5
                elif "Step 7:" in line or "Committing to Production" in line or "Production vector store" in line:
                    new_step = 5

                if new_step > current_step:
                    current_step = new_step

                # Send SSE log event
                yield _sse("log", {"line": line, "step": current_step})

            # Wait for process to finish
            return_code = await asyncio.to_thread(process.wait)

            # Read report if available
            report_path = STAGING_REPORTS / f"{doc_id}_report.json"
            report = {}
            if report_path.exists() and report_path.stat().st_mtime >= started_at:
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                except Exception:
                    pass
            elif report_path.exists():
                logger.warning(f"Ignoring stale ingestion report for {doc_id}: {report_path}")

            if return_code == 0:
                yield _sse("complete", {
                    "status": "success",
                    "doc_id": doc_id,
                    "num_chunks": report.get("num_chunks", 0),
                    "faiss_total": report.get("faiss_current_ntotal", 0) + report.get("num_chunks", 0),
                    "embedding_dim": report.get("embedding_dim", 0),
                })
            else:
                yield _sse("pipeline_error", {
                    "status": "failed",
                    "doc_id": doc_id,
                    "errors": report.get("errors", [f"Process exited with code {return_code}"]),
                })

        except Exception as e:
            error_detail = f"{type(e).__name__}: {e!r}"
            logger.error(f"SSE streaming error: {error_detail}\n{traceback.format_exc()}")
            yield _sse("pipeline_error", {
                "status": "failed",
                "doc_id": doc_id,
                "errors": [error_detail, traceback.format_exc()],
            })

        finally:
            if process and process.returncode is None:
                try:
                    process.kill()
                except Exception:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
