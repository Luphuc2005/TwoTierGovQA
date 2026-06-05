"""
run_pipeline.py — CHẠY PIPELINE ĐƠN GIẢN
==========================================
Chỉ cần sửa config.py rồi chạy:

    python run_pipeline.py

Không cần nhớ tham số CLI nào cả.
"""

import sys
import json
import time
from pathlib import Path

# ── Load config ──
import config as cfg

# Add parent path
sys.path.append(str(Path(__file__).parent.parent))

from generation.llm_client import (
    LLMClient, LLMConfig, LLMBackend, LLMMode, select_local_model_for_vram
)
from generation.run_generation import (
    GenerationPipeline, LegalRetriever, PipelineLogger,
    load_reranked_jsonl, reranked_record_to_chunks,
    load_dev_jsonl, GEN_EVAL_DIR, EVAL_QA_FILE
)
from generation.rag_contract import LLMTier
from generation.retrieval_rerank import RetrievalReranker


def build_llm_client() -> LLMClient:
    """Tạo LLM client từ config.py."""
    
    backend_map = {
        "huggingface": LLMBackend.HUGGINGFACE,
        "qwen": LLMBackend.QWEN,
        "gemini": LLMBackend.GEMINI,
        "openai": LLMBackend.OPENAI,
        "openrouter": LLMBackend.OPENROUTER,
        "llama_cpp": LLMBackend.LLAMA_CPP,
        "placeholder": LLMBackend.PLACEHOLDER,
    }
    
    backend = backend_map.get(cfg.BACKEND, LLMBackend.PLACEHOLDER)
    
    # Xác định model name
    model_name = cfg.MODEL
    model_path = None
    
    if cfg.BACKEND == "huggingface":
        if model_name == "auto" or not model_name:
            model_name = select_local_model_for_vram(reserved_vram_gb=1.0)
        model_path = model_name  # HuggingFace dùng model_path = model_name
    elif cfg.BACKEND == "qwen":
        if not model_name or model_name == "auto":
            model_name = "qwen/qwen3-30b-a3b:free"
    elif cfg.BACKEND == "gemini":
        if not model_name or model_name == "auto":
            model_name = "gemini-2.5-flash"
    elif cfg.BACKEND == "openai":
        if not model_name or model_name == "auto":
            model_name = "gpt-4o-mini"
    
    # API key
    api_key = cfg.API_KEY or None
    if not api_key:
        import os
        if cfg.BACKEND == "qwen":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        elif cfg.BACKEND == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
        elif cfg.BACKEND == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif cfg.BACKEND == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
    
    llm_config = LLMConfig(
        backend=backend,
        mode=LLMMode.DEV if cfg.PIPELINE_MODE == "dev" else LLMMode.DEMO,
        model_name=model_name,
        model_path=model_path,
        max_tokens=cfg.MAX_TOKENS,
        temperature=cfg.TEMPERATURE,
        top_p=cfg.TOP_P,
        context_length=cfg.CONTEXT_LENGTH,
        api_key=api_key,
    )
    
    return LLMClient(llm_config)


def build_reranker() -> RetrievalReranker:
    """Tạo RetrievalReranker từ config.py."""
    from pathlib import Path as _P

    bi_path    = _P(cfg.BI_MODEL_PATH)    if cfg.BI_MODEL_PATH    else None
    ce_path    = _P(cfg.CE_MODEL_PATH)    if cfg.CE_MODEL_PATH    else None
    faiss_path = _P(cfg.FAISS_INDEX_PATH) if cfg.FAISS_INDEX_PATH else None

    return RetrievalReranker(
        bi_model_path=bi_path,
        ce_model_path=ce_path,
        faiss_index_path=faiss_path,
        ce_batch_size=cfg.CE_BATCH_SIZE,
    )


def run():
    """Chạy pipeline theo config."""
    
    print("=" * 60)
    print(" Legal RAG Pipeline — Simple Runner")
    print("=" * 60)
    print(f" Backend:  {cfg.BACKEND}")
    print(f" Model:    {cfg.MODEL}")
    print(f" Mode:     {cfg.RUN_MODE}")
    print(f" Pipeline: {cfg.PIPELINE_MODE}")
    print("=" * 60)
    
    # ── Build LLM client ──
    client = build_llm_client()
    
    # Dùng cùng 1 client cho cả 2 tier (đơn giản)
    local_client = client
    api_client = client
    
    # ── Logger ──
    GEN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = cfg.LOG_DIR or str(GEN_EVAL_DIR / "logs")
    logger = PipelineLogger(log_dir=log_dir, append=cfg.LOG_APPEND)
    
    # ── Retriever ──
    if cfg.RUN_MODE == "reranked":
        retriever = None  # Không cần retriever khi dùng reranked input
    else:
        retriever = LegalRetriever()
    
    # ── Pipeline ──
    pipeline = GenerationPipeline(
        retriever=retriever,
        local_client=local_client,
        api_client=api_client,
        logger=logger,
        mode=LLMMode.DEV if cfg.PIPELINE_MODE == "dev" else LLMMode.DEMO,
        local_first=cfg.LOCAL_FIRST,
        ce_sigmoid_mode=(cfg.RUN_MODE == "reranked"),
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHẠY THEO CHẾ ĐỘ
    # ═══════════════════════════════════════════════════════════════════════
    
    if cfg.RUN_MODE == "query":
        _run_single_query(pipeline, logger)
    elif cfg.RUN_MODE == "reranked":
        _run_reranked(pipeline, logger)
    elif cfg.RUN_MODE == "eval":
        _run_eval(pipeline, logger)
    elif cfg.RUN_MODE == "interactive":
        _run_interactive(pipeline, logger)
    else:
        print(f"[ERROR] RUN_MODE không hợp lệ: {cfg.RUN_MODE}")
        print("  Chọn: 'reranked', 'query', 'eval', 'interactive'")


def _run_single_query(pipeline, logger):
    """Chạy 1 câu hỏi:  Retrieval+Rerank → top-K → Generation."""
    query = cfg.QUERY
    if not query:
        print("[ERROR] Chưa đặt QUERY trong config.py")
        return
    
    print(f"\n📝 Query: {query}\n")

    # ── Step 1: Retrieval + Rerank (v6 pipeline) ──
    print("─" * 50)
    print("  STEP 1: Retrieval + Rerank")
    print("─" * 50)
    reranker = build_reranker()
    top_k = reranker.run(
        query,
        top_n=cfg.TOP_N_RETRIEVE,
        top_k=cfg.TOP_K_RERANK,
    )

    print(f"\n  TOP-{cfg.TOP_K_RERANK} sau rerank:")
    for r in top_k:
        cite = ""
        if r["dieu"]:
            cite = f"Điều {r['dieu']}"
            if r["khoan"]: cite += f", Khoản {r['khoan']}"
        print(f"    [{r['rank']}] CE={r['ce_score']:+.4f}  {cite:20s}  {r['passage'][:60]}...")
    print()

    # Convert top-K → ChunkInfo list cho generation
    from generation.rag_contract import ChunkInfo
    chunks = []
    for r in top_k:
        chunks.append(ChunkInfo(
            chunk_id=r.get("faiss_id", r.get("chunk_index", -1)),
            text=r["passage"],
            score_retrieval=0.0,
            score_rerank=float(r["ce_score"]),
            van_ban=r.get("van_ban", ""),
            chuong=r.get("chuong"),
            dieu=str(r["dieu"]) if r.get("dieu") else None,
            khoan=str(r["khoan"]) if r.get("khoan") else None,
            diem=str(r["diem"]) if r.get("diem") else None,
        ))

    # ── Step 2: Generation ──
    print("─" * 50)
    print("  STEP 2: Generation (LLM)")
    print("─" * 50)
    output, metadata = pipeline.generate_from_reranked(
        query=query,
        chunks=chunks,
        query_id="single_query",
        verbose=cfg.VERBOSE,
    )
    
    print("\n" + "=" * 40)
    tier_str = metadata.get("tier", "?").upper()
    print(f"KẾT QUẢ (Tier: {tier_str}):")
    print("=" * 40)
    
    if output.abstain:
        print(f"[ABSTAIN] {output.reason_detail}")
        if output.clarification_question:
            print(f"❓ {output.clarification_question}")
    else:
        print(f"Trả lời: {output.answer}")
        print(f"\nTrích dẫn:")
        for cit in output.citations:
            print(f"  - {cit.to_str()}")
    
    print(f"\nThời gian: {metadata['timestamps']['total']:.0f}ms")
    logger.close()


def _run_reranked(pipeline, logger):
    """Chạy từ file reranked JSONL."""
    from pathlib import Path as _P
    
    reranked_path = _P(cfg.RERANKED_INPUT)
    if not reranked_path.exists():
        # Thử relative path từ thư mục generation
        reranked_path = Path(__file__).parent / cfg.RERANKED_INPUT
    
    if not reranked_path.exists():
        print(f"[ERROR] Không tìm thấy file: {cfg.RERANKED_INPUT}")
        return
    
    records = load_reranked_jsonl(reranked_path)
    print(f"\n📂 Loaded {len(records)} queries từ {reranked_path.name}")
    
    if cfg.MAX_SAMPLES:
        records = records[:cfg.MAX_SAMPLES]
        print(f"   Giới hạn: {len(records)} queries\n")
    
    tier_counts = {"local": 0, "api": 0, "none": 0}
    citation_hits = 0
    pass_count = 0
    error_count = 0
    n = len(records)
    
    for i, rec in enumerate(records):
        qid = rec.get("id", f"q_{i}")
        query = rec.get("query", "")
        expected = rec.get("expected_citations", [])
        chunks = reranked_record_to_chunks(rec)
        
        if cfg.VERBOSE:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{n}] {qid}: {query[:70]}...")
        
        output, metadata = pipeline.generate_from_reranked(
            query=query, chunks=chunks,
            query_id=qid, verbose=cfg.VERBOSE
        )
        
        # Rate limit
        tier = metadata.get("tier", "none")
        if tier == "api" and i < n - 1:
            time.sleep(1.5)
        
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        gating = metadata.get("decisions", {}).get("gating", {})
        if gating.get("decision") in ("answer", "cautious"):
            pass_count += 1
        
        # Citation hit
        if not output.abstain and output.citations:
            for cit in output.citations:
                for ec in expected:
                    if cit.matches(ec):
                        citation_hits += 1
                        if logger.entries:
                            logger.entries[-1].citation_hit = True
                        break
        
        if output.raw_response and not output.answer:
            error_count += 1
        
        if not cfg.VERBOSE and (i + 1) % 10 == 0:
            print(f"  ... processed {i+1}/{n} queries")
    
    # Summary
    print("\n" + "=" * 60)
    print("  KẾT QUẢ TỔNG HỢP")
    print("=" * 60)
    print(f"  File:              {reranked_path.name}")
    print(f"  Backend:           {cfg.BACKEND} / {cfg.MODEL}")
    print(f"  Tổng queries:      {n}")
    if n:
        print(f"  Citation Hit Rate: {citation_hits/n:.2%}")
        print(f"  Pass Rate:         {pass_count/n:.2%}")
        print(f"  Error Rate:        {error_count/n:.2%}")
    print()
    print("  ┌─ TIER ROUTING ─────────────────────────┐")
    for tier_name, count in tier_counts.items():
        label = {"local": "T1 LOCAL (free)", "api": "T2 API (paid)", "none": "NONE (skip)  "}
        bar = "█" * int(count / n * 30) if n else ""
        if n:
            print(f"  │ {label.get(tier_name, tier_name):16s}  {count:4d}  {count/n:.1%}  {bar}")
    if n:
        savings = (tier_counts.get("local", 0) + tier_counts.get("none", 0)) / n * 100
        print(f"  │ API savings: {savings:.1f}%")
    print("  └─────────────────────────────────────────┘")
    print(f"\n  📋 Log: {logger.jsonl_path}")
    
    logger.close()


def _run_eval(pipeline, logger):
    """Chạy evaluation trên goldset."""
    goldset_path = EVAL_QA_FILE
    if not goldset_path.exists():
        print(f"[ERROR] Goldset not found: {goldset_path}")
        return
    
    samples = load_dev_jsonl(goldset_path)
    print(f"\n📊 Loaded {len(samples)} samples từ {goldset_path.name}")
    
    if cfg.MAX_SAMPLES:
        samples = samples[:cfg.MAX_SAMPLES]
    
    for i, sample in enumerate(samples):
        if cfg.VERBOSE:
            print(f"\n[{i+1}/{len(samples)}] {sample.query[:60]}...")
        output, metadata = pipeline.generate(sample.query, verbose=cfg.VERBOSE)
        tier = metadata.get("tier", "none")
        if tier == "api" and i < len(samples) - 1:
            time.sleep(1.5)
    
    logger.close()
    print(f"\n✅ Evaluation xong! Log: {logger.jsonl_path}")


def _run_interactive(pipeline, logger):
    """Chế độ hỏi đáp tương tác với retrieval+rerank."""
    print("\n💬 Interactive mode (retrieval+rerank+generation). Gõ 'quit' để thoát.\n")
    
    # Load reranker 1 lần
    reranker = build_reranker()
    from generation.rag_contract import ChunkInfo
    
    while True:
        try:
            query = input("❓ Câu hỏi: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query:
                continue
            
            # Retrieve + Rerank
            top_k = reranker.run(query, top_n=cfg.TOP_N_RETRIEVE, top_k=cfg.TOP_K_RERANK)
            print(f"  🔍 Top-{len(top_k)} passages retrieved")
            for r in top_k:
                print(f"     [{r['rank']}] CE={r['ce_score']:+.4f}  {r['passage'][:50]}...")
            
            # Convert → ChunkInfo
            chunks = [
                ChunkInfo(
                    chunk_id=r.get("faiss_id", r.get("chunk_index", -1)),
                    text=r["passage"],
                    score_retrieval=0.0,
                    score_rerank=float(r["ce_score"]),
                    van_ban=r.get("van_ban", ""),
                    chuong=r.get("chuong"),
                    dieu=str(r["dieu"]) if r.get("dieu") else None,
                    khoan=str(r["khoan"]) if r.get("khoan") else None,
                    diem=str(r["diem"]) if r.get("diem") else None,
                )
                for r in top_k
            ]
            
            # Generate
            output, metadata = pipeline.generate_from_reranked(
                query=query, chunks=chunks,
                query_id="interactive", verbose=cfg.VERBOSE,
            )
            
            tier_str = metadata.get("tier", "?").upper()
            print("\n" + "-" * 40)
            if output.abstain:
                print(f"[{output.decision.value.upper()}] {output.reason_detail}")
            else:
                print(f"[Tier: {tier_str}] {output.answer}")
                if output.citations:
                    print(f"📎 {', '.join(c.to_str() for c in output.citations)}")
            print(f"[{metadata['timestamps']['total']:.0f}ms | Tier: {tier_str}]")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\nThoát...")
            break
    
    logger.close()


if __name__ == "__main__":
    run()
