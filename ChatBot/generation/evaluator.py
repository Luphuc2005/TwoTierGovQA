"""
evaluator.py — E2E Generation Evaluation trên goldset

Metrics:
1. Citation Correctness: trích dẫn có đúng điều/khoản không
2. Answer Supported Rate: câu trả lời có được support bởi context không
3. Abstain Precision: abstain có đúng lúc không
4. Latency: retrieval + rerank + LLM

Pipeline: retrieve → rerank → generate → parse citations → evaluate
"""

import json
import time
import csv
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

from .rag_contract import (
    RAGInput, RAGOutput, ChunkInfo, Citation,
    DecisionType, AbstainReason, LLMTier,
    create_rag_input, parse_rag_output, compute_citation_correctness
)
from .context_builder import ContextBuilder, build_context_for_generation
from .prompt_templates import LegalPromptBuilder, PromptConfig
from .gating import GatingStrategy, GatingDecision, GatingConfig
from .llm_client import LLMClient, LLMConfig


@dataclass
class EvalSample:
    """Một sample trong goldset."""
    
    query_id: str
    query: str
    expected_citations: List[Dict]
    
    # Optional ground truth
    expected_answer: Optional[str] = None
    is_answerable: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvalResult:
    """Kết quả evaluation cho một sample."""
    
    query_id: str
    query: str
    
    # Retrieved info
    retrieved_chunk_ids: List[int] = field(default_factory=list)
    reranked_chunk_ids: List[int] = field(default_factory=list)
    reranked_scores: List[float] = field(default_factory=list)
    
    # Gating
    gating_decision: Optional[str] = None
    gating_reason: Optional[str] = None
    tier: str = ""                  # local / api / none
    
    # Generation
    generated_answer: Optional[str] = None
    parsed_citations: List[Dict] = field(default_factory=list)
    abstain: bool = False
    abstain_reason: Optional[str] = None
    
    # Metrics
    citation_hit: bool = False
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0
    answer_supported: Optional[bool] = None  # Requires human eval or LLM-as-judge
    
    # Latency breakdown
    latency_retrieve_ms: float = 0.0
    latency_rerank_ms: float = 0.0
    latency_llm_ms: float = 0.0
    latency_total_ms: float = 0.0
    
    # Raw data
    raw_llm_response: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class EvalSummary:
    """Summary của evaluation run."""
    
    # Run info
    run_id: str
    timestamp: str
    total_samples: int
    
    # Overall metrics
    citation_hit_rate: float = 0.0
    avg_citation_precision: float = 0.0
    avg_citation_recall: float = 0.0
    avg_citation_f1: float = 0.0
    
    # Gating metrics
    pass_rate: float = 0.0
    abstain_rate: float = 0.0
    ask_back_rate: float = 0.0
    cautious_rate: float = 0.0
    
    # Answer metrics
    answer_supported_rate: Optional[float] = None  # Requires human eval
    
    # Latency
    avg_latency_retrieve_ms: float = 0.0
    avg_latency_rerank_ms: float = 0.0
    avg_latency_llm_ms: float = 0.0
    avg_latency_total_ms: float = 0.0
    
    # Errors
    error_rate: float = 0.0
    
    # Config info
    llm_backend: str = ""
    gating_config: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class GenerationEvaluator:
    """
    End-to-End Generation Evaluator.
    
    Pipeline:
    1. Load goldset
    2. For each sample: retrieve → rerank → gate → generate → parse → evaluate
    3. Compute aggregate metrics
    4. Save results
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        gating_strategy: Optional[GatingStrategy] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompt_builder: Optional[LegalPromptBuilder] = None,
        output_dir: Optional[Path] = None
    ):
        self.llm_client = llm_client or LLMClient()
        self.gating = gating_strategy or GatingStrategy()
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or LegalPromptBuilder()
        self.output_dir = output_dir or Path("outputs/eval")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_single(
        self,
        sample: EvalSample,
        chunks: List[ChunkInfo],
        verbose: bool = False
    ) -> EvalResult:
        """
        Evaluate một sample.
        
        Args:
            sample: EvalSample với query và expected_citations
            chunks: Chunks đã được rerank (từ retriever + cross-encoder)
            verbose: Print debug info
        """
        
        result = EvalResult(
            query_id=sample.query_id,
            query=sample.query
        )
        
        start_total = time.time()
        
        try:
            # Extract reranked info
            result.reranked_chunk_ids = [c.chunk_id for c in chunks]
            result.reranked_scores = [c.score_rerank for c in chunks]
            
            # ─────────────────────────────────────────────────────────────────
            # Step 1: Gating
            # ─────────────────────────────────────────────────────────────────
            
            rag_input = RAGInput(
                question=sample.query,
                top_k_chunks=chunks,
                policy=self.gating.config if hasattr(self.gating, 'config') else None
            )
            
            gating_decision = self.gating.evaluate(rag_input)
            result.gating_decision = gating_decision.decision.value
            result.gating_reason = gating_decision.reason_detail
            result.tier = gating_decision.tier.value if hasattr(gating_decision, 'tier') else ""
            
            if verbose:
                print(f"  [Gating] {gating_decision.decision.value}: {gating_decision.reason_detail}")
            
            # ─────────────────────────────────────────────────────────────────
            # Step 2: Generate (if not abstain)
            # ─────────────────────────────────────────────────────────────────
            
            if not self.gating.should_generate(gating_decision):
                # Gating decided to abstain/ask_back
                result.abstain = True
                result.abstain_reason = gating_decision.reason_detail
                
                if gating_decision.decision == DecisionType.ASK_BACK:
                    result.generated_answer = gating_decision.clarification_question
            else:
                # Build context
                processed_chunks, context_string = self.context_builder.build(
                    chunks=chunks,
                    query=sample.query,
                    max_chunks=5
                )
                
                # Build prompt
                prompt = self.prompt_builder.build_full_prompt(
                    question=sample.query,
                    context=context_string
                )
                
                # Generate
                start_llm = time.time()
                llm_response = self.llm_client.generate(prompt)
                result.latency_llm_ms = (time.time() - start_llm) * 1000
                
                result.raw_llm_response = llm_response.text
                
                if llm_response.error:
                    result.error = llm_response.error
                else:
                    # Parse response
                    rag_output = parse_rag_output(llm_response.text)
                    
                    result.generated_answer = rag_output.answer
                    result.abstain = rag_output.abstain
                    result.abstain_reason = rag_output.reason_detail
                    result.parsed_citations = [c.to_dict() for c in rag_output.citations]
                    
                    if verbose:
                        print(f"  [Generated] Abstain={rag_output.abstain}, Citations={len(rag_output.citations)}")
            
            # ─────────────────────────────────────────────────────────────────
            # Step 3: Evaluate citations
            # ─────────────────────────────────────────────────────────────────
            
            if not result.abstain and result.parsed_citations:
                # Convert parsed citations to Citation objects
                output_citations = []
                for cit_dict in result.parsed_citations:
                    output_citations.append(Citation(
                        van_ban=cit_dict.get("van_ban", ""),
                        dieu=cit_dict.get("dieu"),
                        khoan=cit_dict.get("khoan"),
                        diem=cit_dict.get("diem")
                    ))
                
                rag_output_for_eval = RAGOutput(citations=output_citations)
                
                # Compute citation metrics
                citation_metrics = compute_citation_correctness(
                    rag_output_for_eval,
                    sample.expected_citations
                )
                
                result.citation_precision = citation_metrics["precision"]
                result.citation_recall = citation_metrics["recall"]
                result.citation_f1 = citation_metrics["f1"]
                result.citation_hit = citation_metrics["recall"] > 0
            else:
                # Check if any reranked chunk matches expected citation
                for chunk in chunks[:5]:
                    for ec in sample.expected_citations:
                        if self._chunk_matches_citation(chunk, ec):
                            result.citation_hit = True
                            break
            
            result.latency_total_ms = (time.time() - start_total) * 1000
            
        except Exception as e:
            result.error = str(e)
            result.latency_total_ms = (time.time() - start_total) * 1000
        
        return result
    
    def _chunk_matches_citation(self, chunk: ChunkInfo, citation: Dict) -> bool:
        """Check if a chunk matches expected citation."""
        
        # Match by chunk_index
        if citation.get("chunk_index") is not None:
            if chunk.chunk_id == citation.get("chunk_index"):
                return True
        
        # Match by dieu/khoan/diem
        if chunk.dieu and citation.get("dieu"):
            if str(chunk.dieu) == str(citation.get("dieu")):
                if citation.get("khoan"):
                    if str(chunk.khoan) == str(citation.get("khoan")):
                        return True
                else:
                    return True
        
        return False
    
    def evaluate_batch(
        self,
        samples: List[EvalSample],
        retriever_fn,  # Function: (query) -> List[ChunkInfo]
        verbose: bool = False,
        max_samples: Optional[int] = None
    ) -> Tuple[List[EvalResult], EvalSummary]:
        """
        Evaluate batch của samples.
        
        Args:
            samples: List of EvalSample
            retriever_fn: Function that takes query and returns reranked chunks
            verbose: Print progress
            max_samples: Limit number of samples
        
        Returns:
            (results, summary)
        """
        
        if max_samples:
            samples = samples[:max_samples]
        
        results = []
        
        for i, sample in enumerate(samples):
            if verbose:
                print(f"[{i+1}/{len(samples)}] Evaluating: {sample.query[:50]}...")
            
            # Get reranked chunks
            chunks = retriever_fn(sample.query)
            
            # Evaluate
            result = self.evaluate_single(sample, chunks, verbose=verbose)
            results.append(result)
        
        # Compute summary
        summary = self._compute_summary(results)
        
        return results, summary
    
    def _compute_summary(self, results: List[EvalResult]) -> EvalSummary:
        """Compute aggregate metrics."""
        
        n = len(results)
        if n == 0:
            return EvalSummary(
                run_id="empty",
                timestamp=datetime.now().isoformat(),
                total_samples=0
            )
        
        # Citation metrics
        citation_hits = sum(1 for r in results if r.citation_hit)
        avg_precision = sum(r.citation_precision for r in results) / n
        avg_recall = sum(r.citation_recall for r in results) / n
        avg_f1 = sum(r.citation_f1 for r in results) / n
        
        # Gating metrics
        pass_count = sum(1 for r in results if r.gating_decision == "answer")
        abstain_count = sum(1 for r in results if r.gating_decision == "abstain")
        ask_back_count = sum(1 for r in results if r.gating_decision == "ask_back")
        cautious_count = sum(1 for r in results if r.gating_decision == "cautious")
        
        # Latency
        avg_llm = sum(r.latency_llm_ms for r in results) / n
        avg_total = sum(r.latency_total_ms for r in results) / n
        
        # Errors
        error_count = sum(1 for r in results if r.error)
        
        return EvalSummary(
            run_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            total_samples=n,
            citation_hit_rate=round(citation_hits / n, 4),
            avg_citation_precision=round(avg_precision, 4),
            avg_citation_recall=round(avg_recall, 4),
            avg_citation_f1=round(avg_f1, 4),
            pass_rate=round(pass_count / n, 4),
            abstain_rate=round(abstain_count / n, 4),
            ask_back_rate=round(ask_back_count / n, 4),
            cautious_rate=round(cautious_count / n, 4),
            avg_latency_llm_ms=round(avg_llm, 2),
            avg_latency_total_ms=round(avg_total, 2),
            error_rate=round(error_count / n, 4),
            llm_backend=self.llm_client.config.backend.value if self.llm_client else "unknown"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE & LOAD RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_results(
        self,
        results: List[EvalResult],
        summary: EvalSummary,
        prefix: str = "generation_eval"
    ):
        """Save evaluation results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSONL
        results_path = self.output_dir / f"{prefix}_results_{timestamp}.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        
        # Save summary as JSON
        summary_path = self.output_dir / f"{prefix}_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save summary as Markdown
        md_path = self.output_dir / f"{prefix}_summary_{timestamp}.md"
        self._save_summary_md(summary, results, md_path)
        
        # Save as CSV for easy analysis
        csv_path = self.output_dir / f"{prefix}_metrics_{timestamp}.csv"
        self._save_metrics_csv(results, csv_path)
        
        print(f"Results saved to:")
        print(f"  - {results_path}")
        print(f"  - {summary_path}")
        print(f"  - {md_path}")
        print(f"  - {csv_path}")
        
        return {
            "results": results_path,
            "summary": summary_path,
            "markdown": md_path,
            "csv": csv_path
        }
    
    def _save_summary_md(
        self,
        summary: EvalSummary,
        results: List[EvalResult],
        path: Path
    ):
        """Save summary as Markdown."""
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Generation Evaluation Report\n\n")
            f.write(f"**Run ID:** {summary.run_id}\n")
            f.write(f"**Timestamp:** {summary.timestamp}\n")
            f.write(f"**LLM Backend:** {summary.llm_backend}\n\n")
            
            f.write("## Citation Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Citation Hit Rate | {summary.citation_hit_rate:.2%} |\n")
            f.write(f"| Avg Precision | {summary.avg_citation_precision:.4f} |\n")
            f.write(f"| Avg Recall | {summary.avg_citation_recall:.4f} |\n")
            f.write(f"| Avg F1 | {summary.avg_citation_f1:.4f} |\n\n")
            
            f.write("## Gating Metrics\n\n")
            f.write("| Decision | Rate |\n")
            f.write("|----------|------|\n")
            f.write(f"| Pass (Answer) | {summary.pass_rate:.2%} |\n")
            f.write(f"| Cautious | {summary.cautious_rate:.2%} |\n")
            f.write(f"| Abstain | {summary.abstain_rate:.2%} |\n")
            f.write(f"| Ask Back | {summary.ask_back_rate:.2%} |\n\n")
            
            f.write("## Latency\n\n")
            f.write("| Stage | Avg (ms) |\n")
            f.write("|-------|----------|\n")
            f.write(f"| LLM | {summary.avg_latency_llm_ms:.1f} |\n")
            f.write(f"| Total | {summary.avg_latency_total_ms:.1f} |\n\n")
            
            f.write("## Sample Results\n\n")
            f.write("| Query | Decision | Citation Hit | Answer |\n")
            f.write("|-------|----------|--------------|--------|\n")
            
            for r in results[:20]:  # Show first 20
                query_short = r.query[:40] + "..." if len(r.query) > 40 else r.query
                answer_short = (r.generated_answer or "N/A")[:30] + "..." if r.generated_answer else "N/A"
                f.write(f"| {query_short} | {r.gating_decision} | {'✓' if r.citation_hit else '✗'} | {answer_short} |\n")
    
    def _save_metrics_csv(self, results: List[EvalResult], path: Path):
        """Save metrics as CSV."""
        
        fieldnames = [
            "query_id", "query", "gating_decision", "tier", "citation_hit",
            "citation_precision", "citation_recall", "citation_f1",
            "abstain", "latency_llm_ms", "latency_total_ms", "error"
        ]
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                writer.writerow({
                    "query_id": r.query_id,
                    "query": r.query,
                    "gating_decision": r.gating_decision,
                    "tier": r.tier,
                    "citation_hit": r.citation_hit,
                    "citation_precision": r.citation_precision,
                    "citation_recall": r.citation_recall,
                    "citation_f1": r.citation_f1,
                    "abstain": r.abstain,
                    "latency_llm_ms": r.latency_llm_ms,
                    "latency_total_ms": r.latency_total_ms,
                    "error": r.error
                })


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDSET LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_goldset(path: Path) -> List[EvalSample]:
    """Load goldset from JSONL file."""
    
    samples = []
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                sample = EvalSample(
                    query_id=data.get("id", f"q_{i}"),
                    query=data.get("query", data.get("question", "")),
                    expected_citations=data.get("expected_citations", []),
                    expected_answer=data.get("expected_answer"),
                    is_answerable=data.get("is_answerable", True)
                )
                samples.append(sample)
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {i+1}")
                continue
    
    return samples


def create_goldset_from_eval_qa(eval_qa_path: Path) -> List[EvalSample]:
    """Create goldset from existing eval_qa.jsonl format."""
    
    samples = []
    
    with open(eval_qa_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Convert format
                expected_citations = []
                for ec in data.get("expected_citations", []):
                    expected_citations.append({
                        "van_ban": ec.get("van_ban", ""),
                        "chuong": ec.get("chuong"),
                        "dieu": ec.get("dieu"),
                        "khoan": ec.get("khoan"),
                        "diem": ec.get("diem"),
                        "chunk_index": ec.get("chunk_index")
                    })
                
                sample = EvalSample(
                    query_id=data.get("id", f"eval_{i}"),
                    query=data.get("query", ""),
                    expected_citations=expected_citations
                )
                samples.append(sample)
                
            except json.JSONDecodeError:
                continue
    
    return samples
