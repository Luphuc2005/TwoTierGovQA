# Scripts — Ingestion & Maintenance Tools

## Danh sách scripts

### `ingest_new_pdf.py` — Nạp PDF mới vào production index
```bash
# Môi trường: conda activate ChatBot (cần torch, paddleocr, faiss, ...)

# Chạy thử (không ghi production):
python scripts/ingest_new_pdf.py --pdf path/to/document.pdf --dry-run

# Ghi thật vào production:
python scripts/ingest_new_pdf.py --pdf path/to/document.pdf --commit
```

Pipeline 7 bước:
1. Validate PDF (hash, size, duplicate check)
2. Extract text (pdfplumber nếu có text; PaddleOCR nếu file quét)
3. Merge & clean text → full_text.txt
4. Parse cấu trúc văn bản luật (LawParser) → JSON
5. Chunking → từng điều/khoản/điểm
6. Embedding (bi_bge_m3_ft model) → vectors
7. Commit vào FAISS + Metadata + BM25 (có backup + rollback)

---

### `delete_doc.py` — Xoá tài liệu khỏi index
```bash
python scripts/delete_doc.py --doc-id doc_<sha256_prefix>
```

---

### `test_ingested_doc.py` — Kiểm tra tài liệu đã nạp
```bash
python scripts/test_ingested_doc.py
```

---

## Khởi tạo index trống (lần đầu)

Nếu `vector_data/production/` chưa có file, chạy lệnh sau để tạo index trống:

```bash
conda activate ChatBot
python -c "
import faiss, json, pickle, numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

prod = Path('vector_data/production')
prod.mkdir(parents=True, exist_ok=True)

index = faiss.IndexFlatIP(1024)
faiss.write_index(index, str(prod / 'index.faiss'))
with open(prod / 'metadata.json', 'w') as f: json.dump([], f)
bm25_data = {'bm25': BM25Okapi([[]]), 'corpus': [[]]}
with open(prod / 'bm25.pkl', 'wb') as f: pickle.dump(bm25_data, f)
print('Done! Empty index created.')
"
```
