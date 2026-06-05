# rag_pipeline/

Thư mục phát triển RAG pipeline theo hướng **ablation study có kiểm soát**.

## Cấu Trúc

```
rag_pipeline/
├── pipeline_utils.py          # Shared utilities (corpus, metrics, output format)
├── notebooks/
│   ├── v1_bm25_baseline.ipynb         # V1: BM25 (no rerank)
│   ├── v2_1_phobert.ipynb             # V2.1: PhoBERT bi-encoder
│   ├── v2_2_vietnam_hf.ipynb          # V2.2: Vietnam_legal_HF (off-shelf)
│   ├── v2_3_bge_m3.ipynb              # V2.3: bge-m3 (multilingual SOTA)
│   ├── v3_1_msmarco_ce.ipynb          # V3.1: ms-marco MiniLM CE
│   ├── v3_2_phobert_ce.ipynb          # V3.2: PhoBERT CE (đã train)
│   ├── v3_3_bge_reranker.ipynb        # V3.3: bge-reranker-v2-m3
│   ├── v4_finetune_bi.ipynb           # V4: Fine-tune best bi-encoder
│   ├── v5_hard_neg_ce.ipynb           # V5: Hard negative mining (CE)
│   ├── v6_hybrid_bm25_dense.ipynb     # V6: Hybrid BM25 + Dense (RRF)
│   ├── v7_1_query_only.ipynb          # V7.1: Query gốc (baseline HyDE)
│   ├── v7_2_hyde_only.ipynb           # V7.2: HyDE replace query
│   └── v7_3_hyde_combine.ipynb        # V7.3: HyDE + query gốc (combine)
└── outputs/
    ├── eval/
    │   ├── pipeline_results_v1.jsonl  # Đầu ra cho LLM Generation
    │   ├── metrics_v1.csv
    │   └── ...
    ├── tmp/                           # FAISS indexes
    └── models/                        # Fine-tuned models
```

## Chuẩn Đầu Ra (pipeline_results_vX.jsonl)

Mỗi dòng JSONL là 1 câu hỏi, đảm bảo đủ thông tin cho LLM Generation:

```json
{
  "query": "...",                          // ← input cho LLM
  "top5_reranked": [
    {
      "rank": 1,
      "hit": true,
      "passage": "...",                    // ← context đầy đủ cho LLM
      "van_ban": "NGHỊ ĐỊNH ...",          // ← trích dẫn nguồn
      "dieu": "9",
      "khoan": "5",
      "ce_score": 0.9971
    }
  ],
  "hit@1": 1, "hit@3": 1, "hit@5": 1,
  "mrr": 1.0
}
```

## Lộ Trình

| Version | Kỹ thuật | Status |
|---|---|---|
| V1 | BM25 baseline | ⬜ |
| V2.1 | PhoBERT bi-encoder | ⬜ |
| V2.2 | Vietnam_legal_HF | ⬜ |
| V2.3 | bge-m3 | ⬜ |
| V3.1-3 | MS-Marco/PhoBERT/BGE CE | ⬜ |
| V4 | Fine-tune bi-encoder | ⬜ |
| V5 | Hard negative CE | ⬜ |
| V6 | Hybrid BM25+Dense | ⬜ |
| V7.1-3 | HyDE variants | ⬜ |
