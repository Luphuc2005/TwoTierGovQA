"""
gating.py — Gating Strategy: Khi nào trả lời, khi nào abstain

Với cross-encoder ROC-AUC 0.96, ta có confidence gate rất tốt.

Rules:
1. score_top1 < threshold → ABSTAIN
2. (top1 - top2) < margin → ABSTAIN (không rõ ràng)
3. Query type không match context keywords → ABSTAIN
4. Fallback: ASK_BACK hoặc CAUTIOUS answer
"""

import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .rag_contract import (
    ChunkInfo, RAGInput, RAGOutput, RAGPolicy,
    DecisionType, AbstainReason, Citation, LLMTier
)
from .context_builder import ContextBuilder


@dataclass
class GatingConfig:
    """Cấu hình ngưỡng cho gating."""
    
    # Score thresholds
    threshold_pass: float = 2.0           # Score >= này → PASS
    threshold_abstain: float = 0.5        # Score < này → ABSTAIN
    threshold_cautious: float = 1.0       # Giữa abstain và pass → CAUTIOUS
    
    # Margin threshold (normalized: margin / |score|)
    margin_min: float = 0.003            # Normalized margin < này → ambiguous
    margin_scale: float = 50.0           # Hệ số scale cho sigmoid(norm_margin)
    
    # ═══ 2-TIER ROUTING ═══
    # Tier 1 (LOCAL): score cao + relative margin cao → trả lời nhanh, local
    # Tier 2 (API):   score trung bình / margin thấp / cautious → cần API
    # NONE:           score thấp → abstain/ask_back, không gọi LLM
    tier_local_min_score: float = 6.0     # Score tối thiểu để dùng local
    tier_local_min_margin: float = 0.03   # Normalized margin tối thiểu cho LOCAL (3%)
    tier_api_min_score: float = 0.0       # Score tối thiểu để gọi API
    
    # Keyword coverage
    min_keyword_coverage: float = 0.3     # Overlap từ khóa tối thiểu
    
    # Lexical overlap (thay thế hard keyword check)
    min_lexical_overlap: float = 0.15     # Ngưỡng tối thiểu lexical overlap query↔context
    require_legal_keyword: bool = False   # Đã tắt — dùng lexical overlap thay thế
    
    # Ask-back triggers
    enable_ask_back: bool = True
    ask_back_on_ambiguous: bool = True
    
    # Score type flag
    score_is_sigmoid: bool = False        # True khi scores là post-sigmoid CE (0-1)


class GatingDecision:
    """Kết quả của gating decision."""
    
    def __init__(
        self,
        decision: DecisionType,
        reason: Optional[AbstainReason] = None,
        reason_detail: str = "",
        confidence: float = 0.0,
        clarification_question: Optional[str] = None,
        tier: LLMTier = LLMTier.NONE,
        margin: float = 0.0,
        confidence_final: float = 0.0,
        semantic_similarity: float = 0.0,
        query_type: str = "",
        context_token_length: int = 0,
        lexical_overlap: float = 0.0
    ):
        self.decision = decision
        self.reason = reason
        self.reason_detail = reason_detail
        self.confidence = confidence
        self.clarification_question = clarification_question
        self.tier = tier
        self.margin = margin
        self.confidence_final = confidence_final  # sigmoid(margin)
        self.semantic_similarity = semantic_similarity  # rerank score normalized
        self.query_type = query_type
        self.context_token_length = context_token_length
        self.lexical_overlap = lexical_overlap
    
    def to_dict(self) -> Dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason.value if self.reason else None,
            "reason_detail": self.reason_detail,
            "confidence": self.confidence,
            "confidence_final": round(self.confidence_final, 4),
            "clarification_question": self.clarification_question,
            "tier": self.tier.value,
            "margin": round(self.margin, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "query_type": self.query_type,
            "context_token_length": self.context_token_length,
            "lexical_overlap": round(self.lexical_overlap, 4)
        }


class GatingStrategy:
    """
    Strategy để quyết định PASS / ABSTAIN / ASK_BACK / CAUTIOUS.
    
    Pipeline:
    1. Check score threshold
    2. Check score margin (top1 - top2)
    3. Check keyword coverage
    4. Lexical overlap query↔context (thay thế hard keyword check)
    5. Check query ambiguity
    6. Final: confidence = sigmoid(margin), tier routing
    """
    
    def __init__(self, config: Optional[GatingConfig] = None):
        self.config = config or GatingConfig()
        self.context_builder = ContextBuilder()
    
    # ─── Helpers ─────────────────────────────────────────────────────────────
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function, clamped to avoid overflow."""
        x = max(-10.0, min(10.0, x))
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def _compute_lexical_overlap(query: str, context: str) -> float:
        """
        Tính lexical overlap giữa query và context.
        = |query_tokens ∩ context_tokens| / |query_tokens|
        
        Hỗ trợ Vietnamese compound words (2-gram, 3-gram từ pháp lý).
        """
        import re as _re
        from .context_builder import ContextBuilder
        
        _builder = ContextBuilder()
        q_tokens = set(_builder._extract_keywords(query))
        c_tokens = set(_builder._extract_keywords(context))
        
        if not q_tokens:
            return 1.0  # empty query → no check
        
        # Exact overlap
        overlap = q_tokens & c_tokens
        
        # Partial overlap: nếu compound query token xuất hiện dạng single trong context
        # VD: query "thẩm quyền" vs context chứa "thẩm" + "quyền" riêng lẻ
        context_lower = context.lower()
        for qt in q_tokens - overlap:
            if qt in context_lower:
                overlap.add(qt)
        
        return len(overlap) / len(q_tokens)
    
    def _build_common_fields(
        self, top1_score: float, margin: float, query: str, chunks: List[ChunkInfo]
    ) -> Dict:
        """Tính các trường phụ dùng chung cho mọi GatingDecision."""
        
        if self.config.score_is_sigmoid:
            # ── CE post-sigmoid mode ──
            # Scores are 0-1 (post-sigmoid), margins can be tiny (0.0001) or large (0.99)
            # Use absolute margin directly (not normalized), since score is bounded
            abs_margin = margin  # already in 0-1
            
            # Confidence final: sigmoid scale on absolute margin
            # margin ~0    → sigmoid(-1.5) = 0.18 (low — ambiguous top-k)
            # margin ~0.01 → sigmoid(-0.5) = 0.38
            # margin ~0.5  → sigmoid(+48.5) ≈ 1.0 (clear winner)
            # margin ~0.99 → sigmoid(+97.5) ≈ 1.0
            confidence_final = self._sigmoid(
                abs_margin * self.config.margin_scale - 1.5
            )
            
            # Semantic similarity: score IS already 0-1, use directly
            semantic_similarity = top1_score
        else:
            # ── Raw score mode (CE logits or cosine) ──
            # Normalize margin theo scale của score
            norm_margin = margin / max(abs(top1_score), 1.0)
            confidence_final = self._sigmoid(
                norm_margin * self.config.margin_scale - 1.5
            )
            # Semantic similarity: normalize cross-encoder score → [0,1]
            semantic_similarity = self._sigmoid(top1_score / 5.0)
        
        # Query type
        query_types = self.context_builder.detect_query_type(query)
        query_type = ",".join(query_types) if query_types else "general"
        
        # Context token length estimate
        all_text = " ".join(c.text for c in chunks[:5])
        context_token_length = int(len(all_text) / 1.5)
        
        # Lexical overlap
        lexical_overlap = self._compute_lexical_overlap(query, all_text)
        
        return {
            "confidence_final": confidence_final,
            "semantic_similarity": semantic_similarity,
            "query_type": query_type,
            "context_token_length": context_token_length,
            "lexical_overlap": lexical_overlap,
        }
    
    # ─── Main evaluate ───────────────────────────────────────────────────────
    
    def evaluate(self, rag_input: RAGInput) -> GatingDecision:
        """
        Đánh giá và đưa ra gating decision.
        
        Returns:
            GatingDecision with decision type and reasoning
        """
        
        chunks = rag_input.top_k_chunks
        query = rag_input.question
        
        if not chunks:
            return GatingDecision(
                decision=DecisionType.ABSTAIN,
                reason=AbstainReason.NO_RELEVANT_CHUNK,
                reason_detail="Không tìm thấy chunk liên quan",
                confidence=0.0,
                tier=LLMTier.NONE,
                margin=0.0
            )
        
        # Get scores
        scores = [c.score_rerank for c in chunks]
        top1_score = scores[0] if scores else 0.0
        top2_score = scores[1] if len(scores) > 1 else 0.0
        margin = top1_score - top2_score
        
        # Pre-compute common fields (sigmoid confidence, semantic sim, etc.)
        extras = self._build_common_fields(top1_score, margin, query, chunks)
        
        # ─────────────────────────────────────────────────────────────────────
        # Rule 1: Score threshold check
        # ─────────────────────────────────────────────────────────────────────
        
        if top1_score < self.config.threshold_abstain:
            return GatingDecision(
                decision=DecisionType.ABSTAIN,
                reason=AbstainReason.LOW_CONFIDENCE,
                reason_detail=f"Top-1 score ({top1_score:.2f}) < threshold ({self.config.threshold_abstain})",
                confidence=top1_score,
                tier=LLMTier.NONE,
                margin=margin,
                **extras
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # Rule 2: Margin check
        #   BYPASS khi score >= threshold_pass (CE sigmoid mode):
        #   High CE score = highly relevant context. Low margin = multiple
        #   relevant chunks (not ambiguity).  38%+ of queries have margin < 0.001
        #   but score >= 0.99 — these should PASS, not get blocked.
        # ─────────────────────────────────────────────────────────────────────
        
        skip_margin_check = (
            self.config.score_is_sigmoid and top1_score >= self.config.threshold_pass
        )
        
        if self.config.score_is_sigmoid:
            # CE sigmoid: use absolute margin (scores are bounded 0-1)
            effective_margin = margin
        else:
            # Raw scores: normalize by score magnitude
            effective_margin = margin / max(abs(top1_score), 1.0)
        
        if (not skip_margin_check
                and effective_margin < self.config.margin_min
                and self.config.enable_ask_back):
            if self.config.ask_back_on_ambiguous:
                return GatingDecision(
                    decision=DecisionType.ASK_BACK,
                    reason=AbstainReason.AMBIGUOUS_QUERY,
                    reason_detail=f"Margin ({effective_margin:.4f}) < threshold ({self.config.margin_min}), kết quả không rõ ràng",
                    confidence=top1_score,
                    clarification_question="Bạn có thể cho biết thêm chi tiết về câu hỏi?",
                    tier=LLMTier.NONE,
                    margin=margin,
                    **extras
                )
        
        # ─────────────────────────────────────────────────────────────────────
        # Rule 3: Keyword coverage check
        #   BYPASS khi cross-encoder score cao (>= threshold_pass):
        #   Cross-encoder đã làm semantic matching, score cao = context liên quan.
        #   Tránh false-CAUTIOUS khi query không dấu / paraphrase.
        # ─────────────────────────────────────────────────────────────────────
        
        if top1_score < self.config.threshold_pass:
            coverage_info = self.context_builder.check_keyword_coverage(query, chunks)
            
            if coverage_info["coverage"] < self.config.min_keyword_coverage:
                if self.config.enable_ask_back:
                    return GatingDecision(
                        decision=DecisionType.ASK_BACK,
                        reason=AbstainReason.MISSING_LEGAL_KEYWORD,
                        reason_detail=f"Keyword coverage ({coverage_info['coverage']:.2%}) thấp. Missing: {coverage_info['missing'][:5]}",
                        confidence=top1_score,
                        clarification_question=self._generate_clarification_question(query, coverage_info),
                        tier=LLMTier.NONE,
                        margin=margin,
                        **extras
                    )
                else:
                    return GatingDecision(
                        decision=DecisionType.CAUTIOUS,
                        reason=AbstainReason.MISSING_LEGAL_KEYWORD,
                        reason_detail=f"Keyword coverage thấp: {coverage_info['coverage']:.2%}",
                        confidence=top1_score,
                        tier=LLMTier.API,
                        margin=margin,
                        **extras
                    )
        
        # ─────────────────────────────────────────────────────────────────────
        # Rule 4: Lexical overlap check (thay thế hard keyword check cũ)
        #   Kiểm tra query↔context có chung từ vựng không.
        #   4a) overlap ≈ 0 → context chắc chắn sai domain → CAUTIOUS (API)
        #   4b) overlap thấp + score thấp → CAUTIOUS
        # ─────────────────────────────────────────────────────────────────────
        
        lex_overlap = extras["lexical_overlap"]
        
        # 4a: Near-zero overlap = sai domain (VD: query "xây dựng" → chunk "đình công")
        #     Không tin CE score khi 0 từ trùng — dù CE cho 10+ điểm
        if lex_overlap < 0.05:
            return GatingDecision(
                decision=DecisionType.CAUTIOUS,
                reason=AbstainReason.MISSING_LEGAL_KEYWORD,
                reason_detail=(
                    f"Lexical overlap ≈ 0 ({lex_overlap:.2%}) → context có thể sai domain, "
                    f"route API dù CE score={top1_score:.2f}"
                ),
                confidence=top1_score,
                tier=LLMTier.API,
                margin=margin,
                **extras
            )
        
        # 4b: Overlap thấp + score không cao → route qua API (diễn đạt tốt hơn)
        if lex_overlap < self.config.min_lexical_overlap and top1_score < self.config.tier_local_min_score:
            return GatingDecision(
                decision=DecisionType.CAUTIOUS,
                reason=AbstainReason.MISSING_LEGAL_KEYWORD,
                reason_detail=(
                    f"Lexical overlap thấp ({lex_overlap:.2%}) + "
                    f"score ({top1_score:.2f}) < {self.config.tier_local_min_score} "
                    f"→ route API"
                ),
                confidence=top1_score,
                tier=LLMTier.API,
                margin=margin,
                **extras
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # Rule 5: Query ambiguity check
        # ─────────────────────────────────────────────────────────────────────
        
        needs_clarify, clarify_question = self.context_builder.needs_clarification(query)
        
        if needs_clarify and self.config.enable_ask_back:
            return GatingDecision(
                decision=DecisionType.ASK_BACK,
                reason=AbstainReason.AMBIGUOUS_QUERY,
                reason_detail="Query cần được làm rõ",
                confidence=top1_score,
                clarification_question=clarify_question,
                tier=LLMTier.NONE,
                margin=margin,
                **extras
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # Final decision + Tier routing
        # confidence_final = sigmoid(margin): tổng hợp margin thành [0,1]
        # ─────────────────────────────────────────────────────────────────────
        
        tier = self._determine_tier(top1_score, margin)
        
        if top1_score >= self.config.threshold_pass:
            return GatingDecision(
                decision=DecisionType.ANSWER,
                confidence=top1_score,
                reason_detail=f"Passed all checks → Tier {tier.value} (conf={extras['confidence_final']:.2f})",
                tier=tier,
                margin=margin,
                **extras
            )
        elif top1_score >= self.config.threshold_cautious:
            return GatingDecision(
                decision=DecisionType.CAUTIOUS,
                reason_detail=f"Score ({top1_score:.2f}) cautious → Tier API (conf={extras['confidence_final']:.2f})",
                confidence=top1_score,
                tier=LLMTier.API,
                margin=margin,
                **extras
            )
        else:
            return GatingDecision(
                decision=DecisionType.ABSTAIN,
                reason=AbstainReason.LOW_CONFIDENCE,
                reason_detail=f"Score ({top1_score:.2f}) thấp hơn pass threshold",
                confidence=top1_score,
                tier=LLMTier.NONE,
                margin=margin,
                **extras
            )
    
    def _generate_clarification_question(
        self,
        query: str,
        coverage_info: Dict
    ) -> str:
        """Generate câu hỏi làm rõ thông minh."""
        
        missing = coverage_info.get("missing", [])
        
        # Detect what type of info is missing
        location_keywords = ["tỉnh", "huyện", "xã", "địa phương", "thành phố"]
        time_keywords = ["năm", "tháng", "ngày", "khi", "thời điểm"]
        subject_keywords = ["ai", "đối tượng", "người", "tổ chức"]
        
        missing_lower = [m.lower() for m in missing]
        
        if any(kw in query.lower() for kw in location_keywords):
            return "Bạn đang hỏi về địa phương/khu vực nào cụ thể?"
        elif any(kw in query.lower() for kw in time_keywords):
            return "Bạn đang hỏi về thời điểm nào cụ thể?"
        elif any(kw in missing_lower for kw in subject_keywords):
            return "Bạn có thể cho biết đối tượng cụ thể được không?"
        else:
            return "Bạn có thể mô tả chi tiết hơn về câu hỏi?"
    
    def _determine_tier(self, top1_score: float, margin: float) -> LLMTier:
        """
        Xác định tầng LLM dựa trên score và margin.
        
        Tier 1 (LOCAL): score cao + margin cao → trả lời nhanh, chuẩn
        Tier 2 (API):   score trung bình / margin thấp → cần diễn đạt tốt hơn
        NONE:           score thấp → abstain / ask_back
        """
        if self.config.score_is_sigmoid:
            # CE sigmoid mode: use absolute margin (not normalized)
            # margin >= 0.5 means top-1 is clearly better than top-2
            if (top1_score >= self.config.tier_local_min_score and
                margin >= self.config.tier_local_min_margin):
                return LLMTier.LOCAL
            elif top1_score >= self.config.tier_api_min_score:
                return LLMTier.API
            else:
                return LLMTier.NONE
        else:
            # Raw score mode: use normalized margin
            norm_margin = margin / max(abs(top1_score), 1.0)
            if (top1_score >= self.config.tier_local_min_score and 
                norm_margin >= self.config.tier_local_min_margin):
                return LLMTier.LOCAL
            elif top1_score >= self.config.tier_api_min_score:
                return LLMTier.API
            else:
                return LLMTier.NONE
    
    def should_generate(self, decision: GatingDecision) -> bool:
        """Kiểm tra có nên gọi LLM để generate không."""
        return decision.decision in [DecisionType.ANSWER, DecisionType.CAUTIOUS]
    
    def build_gated_output(self, decision: GatingDecision) -> Optional[RAGOutput]:
        """
        Build RAGOutput nếu gating quyết định không cần LLM.
        
        Returns:
            RAGOutput nếu abstain/ask_back, None nếu cần generate
        """
        
        if decision.decision == DecisionType.ANSWER:
            return None  # Cần LLM generate
        
        if decision.decision == DecisionType.CAUTIOUS:
            return None  # Cần LLM generate với cảnh báo
        
        if decision.decision == DecisionType.ABSTAIN:
            return RAGOutput(
                answer=None,
                citations=[],
                decision=DecisionType.ABSTAIN,
                abstain=True,
                abstain_reason=decision.reason,
                reason_detail=decision.reason_detail,
                confidence_score=decision.confidence
            )
        
        if decision.decision == DecisionType.ASK_BACK:
            return RAGOutput(
                answer=None,
                citations=[],
                decision=DecisionType.ASK_BACK,
                abstain=True,
                abstain_reason=decision.reason,
                reason_detail=decision.reason_detail,
                clarification_question=decision.clarification_question,
                confidence_score=decision.confidence
            )
        
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

class ScoreCalibrator:
    """
    Calibrate cross-encoder scores để có thể so sánh across queries.
    """
    
    def __init__(self):
        self.score_history = []
        self.percentiles = {}
    
    def add_scores(self, scores: List[float]):
        """Add scores để học distribution."""
        self.score_history.extend(scores)
        self._update_percentiles()
    
    def _update_percentiles(self):
        """Update percentile thresholds."""
        if len(self.score_history) < 100:
            return
        
        sorted_scores = sorted(self.score_history)
        n = len(sorted_scores)
        
        self.percentiles = {
            "p10": sorted_scores[int(n * 0.10)],
            "p25": sorted_scores[int(n * 0.25)],
            "p50": sorted_scores[int(n * 0.50)],
            "p75": sorted_scores[int(n * 0.75)],
            "p90": sorted_scores[int(n * 0.90)],
        }
    
    def get_percentile(self, score: float) -> float:
        """Get percentile của một score."""
        if not self.score_history:
            return 0.5
        
        below = sum(1 for s in self.score_history if s < score)
        return below / len(self.score_history)
    
    def get_calibrated_config(self) -> GatingConfig:
        """Get GatingConfig calibrated theo score distribution."""
        if not self.percentiles:
            return GatingConfig()
        
        return GatingConfig(
            threshold_pass=self.percentiles.get("p75", 2.0),
            threshold_abstain=self.percentiles.get("p10", 0.5),
            threshold_cautious=self.percentiles.get("p50", 1.0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def quick_gate(
    chunks: List[ChunkInfo],
    threshold: float = 0.5
) -> Tuple[bool, str]:
    """
    Quick gating check.
    
    Returns:
        (should_generate, reason)
    """
    if not chunks:
        return False, "No chunks available"
    
    top_score = chunks[0].score_rerank
    
    if top_score < threshold:
        return False, f"Top score ({top_score:.2f}) below threshold ({threshold})"
    
    return True, "Passed quick gate"


def compute_gating_metrics(
    decisions: List[GatingDecision],
    ground_truth_answerable: List[bool]
) -> Dict[str, float]:
    """
    Compute gating quality metrics.
    
    Metrics:
    - Abstain precision: abstain đúng lúc
    - Abstain recall: không bỏ sót difficult cases
    - Pass accuracy: pass đúng
    """
    
    if len(decisions) != len(ground_truth_answerable):
        raise ValueError("Decisions and ground truth must have same length")
    
    n = len(decisions)
    if n == 0:
        return {}
    
    # Categorize
    true_positives = 0   # Pass và answerable
    true_negatives = 0   # Abstain và not answerable
    false_positives = 0  # Pass nhưng not answerable (bad!)
    false_negatives = 0  # Abstain nhưng answerable
    
    for decision, is_answerable in zip(decisions, ground_truth_answerable):
        passed = decision.decision in [DecisionType.ANSWER, DecisionType.CAUTIOUS]
        
        if passed and is_answerable:
            true_positives += 1
        elif not passed and not is_answerable:
            true_negatives += 1
        elif passed and not is_answerable:
            false_positives += 1
        else:
            false_negatives += 1
    
    # Calculate metrics
    pass_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    abstain_precision = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
    
    total_abstains = sum(1 for d in decisions if d.decision in [DecisionType.ABSTAIN, DecisionType.ASK_BACK])
    abstain_rate = total_abstains / n
    
    return {
        "pass_precision": round(pass_precision, 4),
        "abstain_precision": round(abstain_precision, 4),
        "abstain_rate": round(abstain_rate, 4),
        "false_positive_rate": round(false_positives / n, 4),
        "true_positive_rate": round(true_positives / n, 4),
    }
