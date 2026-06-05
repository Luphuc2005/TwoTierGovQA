"""
fallback.py — Fallback Strategy cho 36% không hit đúng

2 Fallback an toàn cho Legal QA:

1. ASK-BACK: Hỏi lại 1 câu ngắn khi thiếu thông tin
   - Địa phương?
   - Thời điểm?
   - Đối tượng?

2. CAUTIOUS ANSWER: Tóm tắt "những gì tìm được" + cảnh báo
   - Liệt kê thông tin liên quan
   - Cảnh báo "chưa thấy điều khoản trực tiếp"

TUYỆT ĐỐI TRÁNH: "Suy diễn luật"
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .rag_contract import (
    ChunkInfo, RAGInput, RAGOutput, Citation,
    DecisionType, AbstainReason
)


class FallbackType(Enum):
    """Loại fallback."""
    NONE = "none"                       # Không cần fallback
    ASK_BACK = "ask_back"               # Hỏi thêm thông tin
    CAUTIOUS = "cautious"               # Trả lời thận trọng
    ABSTAIN = "abstain"                 # Từ chối trả lời


@dataclass
class FallbackDecision:
    """Quyết định fallback."""
    
    fallback_type: FallbackType
    reason: str
    
    # Nếu ASK_BACK
    clarification_question: Optional[str] = None
    missing_info_type: Optional[str] = None
    
    # Nếu CAUTIOUS
    cautious_prefix: Optional[str] = None
    cautious_warning: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "fallback_type": self.fallback_type.value,
            "reason": self.reason,
            "clarification_question": self.clarification_question,
            "cautious_prefix": self.cautious_prefix,
            "cautious_warning": self.cautious_warning
        }


class FallbackStrategy:
    """
    Strategy để xử lý 36% query không hit đúng citation.
    
    Nguyên tắc:
    - KHÔNG suy diễn luật
    - KHÔNG trả lời nếu không chắc chắn
    - Hỏi lại khi thiếu thông tin
    - Cảnh báo khi trả lời thận trọng
    """
    
    def __init__(self):
        # Patterns cho different missing info types
        self.missing_info_patterns = {
            "location": {
                "triggers": ["địa phương", "tỉnh", "huyện", "xã", "thành phố", "quận", "khu vực"],
                "question": "Bạn đang hỏi về địa phương/khu vực nào cụ thể?",
                "examples": ["tỉnh nào", "huyện nào", "thành phố nào"]
            },
            "time": {
                "triggers": ["thời điểm", "năm", "khi nào", "từ khi", "đến khi", "hiệu lực"],
                "question": "Bạn muốn biết thông tin áp dụng cho thời điểm nào?",
                "examples": ["năm 2025", "trước ngày", "sau khi"]
            },
            "subject": {
                "triggers": ["đối tượng", "ai", "người nào", "tổ chức", "doanh nghiệp", "cá nhân"],
                "question": "Bạn có thể cho biết đối tượng cụ thể (cá nhân/tổ chức/doanh nghiệp)?",
                "examples": ["công dân", "doanh nghiệp", "hộ gia đình"]
            },
            "case": {
                "triggers": ["trường hợp", "tình huống", "nếu", "khi nào"],
                "question": "Bạn có thể mô tả cụ thể trường hợp/tình huống của bạn?",
                "examples": ["trường hợp cụ thể", "tình huống thực tế"]
            },
            "document": {
                "triggers": ["văn bản", "nghị định", "thông tư", "luật", "quy định"],
                "question": "Bạn muốn hỏi về văn bản pháp luật nào cụ thể?",
                "examples": ["nghị định số", "thông tư nào"]
            }
        }
        
        # Cautious answer templates
        self.cautious_templates = {
            "prefix": "Dựa trên các văn bản được cung cấp, tôi tìm thấy một số thông tin có thể liên quan:",
            "warning": "⚠️ **Lưu ý**: Tôi chưa tìm thấy điều khoản trực tiếp quy định về vấn đề bạn hỏi. Thông tin trên chỉ mang tính tham khảo. Vui lòng tham vấn chuyên gia pháp lý để được tư vấn chính xác.",
            "no_info": "Tôi không tìm thấy thông tin liên quan trong các văn bản pháp luật được cung cấp."
        }
    
    def evaluate(
        self,
        query: str,
        chunks: List[ChunkInfo],
        gating_decision: Optional[DecisionType] = None
    ) -> FallbackDecision:
        """
        Đánh giá và quyết định fallback strategy.
        
        Args:
            query: Câu hỏi từ user
            chunks: Chunks đã rerank
            gating_decision: Kết quả từ gating (nếu có)
        
        Returns:
            FallbackDecision
        """
        
        # Check if chunks are empty
        if not chunks:
            return FallbackDecision(
                fallback_type=FallbackType.ABSTAIN,
                reason="Không tìm thấy văn bản liên quan"
            )
        
        # Check scores
        top_score = chunks[0].score_rerank if chunks else 0
        
        # Very low score → Likely need more info
        if top_score < 0.3:
            # Check what info might be missing
            missing_type, question = self._detect_missing_info(query)
            
            if missing_type:
                return FallbackDecision(
                    fallback_type=FallbackType.ASK_BACK,
                    reason=f"Score thấp ({top_score:.2f}), có thể thiếu thông tin về {missing_type}",
                    clarification_question=question,
                    missing_info_type=missing_type
                )
            else:
                return FallbackDecision(
                    fallback_type=FallbackType.ABSTAIN,
                    reason=f"Score quá thấp ({top_score:.2f}), không đủ căn cứ"
                )
        
        # Medium score → Cautious answer
        if top_score < 0.7:
            # Check keyword coverage
            coverage = self._compute_keyword_coverage(query, chunks)
            
            if coverage < 0.3:
                return FallbackDecision(
                    fallback_type=FallbackType.ASK_BACK,
                    reason=f"Keyword coverage thấp ({coverage:.2%})",
                    clarification_question=self._generate_generic_clarification(query)
                )
            else:
                return FallbackDecision(
                    fallback_type=FallbackType.CAUTIOUS,
                    reason=f"Score trung bình ({top_score:.2f}), cần cảnh báo",
                    cautious_prefix=self.cautious_templates["prefix"],
                    cautious_warning=self.cautious_templates["warning"]
                )
        
        # High score → No fallback needed
        return FallbackDecision(
            fallback_type=FallbackType.NONE,
            reason="Đủ tin cậy để trả lời"
        )
    
    def _detect_missing_info(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect what type of information might be missing."""
        
        query_lower = query.lower()
        
        for info_type, patterns in self.missing_info_patterns.items():
            # Check if query mentions the info type but seems incomplete
            for trigger in patterns["triggers"]:
                if trigger in query_lower:
                    # Check if specific value is missing
                    if not self._has_specific_value(query_lower, info_type):
                        return info_type, patterns["question"]
        
        # Check if query is too short/vague
        if len(query.split()) < 5:
            return "detail", "Bạn có thể mô tả chi tiết hơn về câu hỏi?"
        
        return None, None
    
    def _has_specific_value(self, query: str, info_type: str) -> bool:
        """Check if query has specific value for an info type."""
        
        if info_type == "location":
            # Check for specific location names
            location_patterns = r'\b(hà nội|hồ chí minh|đà nẵng|hải phòng|cần thơ|tỉnh \w+|huyện \w+|quận \w+)\b'
            return bool(re.search(location_patterns, query, re.IGNORECASE))
        
        elif info_type == "time":
            # Check for specific time
            time_patterns = r'\b(năm \d{4}|ngày \d{1,2}|\d{1,2}/\d{1,2}/\d{4}|từ năm|đến năm)\b'
            return bool(re.search(time_patterns, query, re.IGNORECASE))
        
        elif info_type == "subject":
            # Check for specific subject
            subject_patterns = r'\b(công dân|doanh nghiệp|hộ gia đình|cá nhân|tổ chức|công ty|cơ quan)\b'
            return bool(re.search(subject_patterns, query, re.IGNORECASE))
        
        return True  # Default: assume has value
    
    def _compute_keyword_coverage(self, query: str, chunks: List[ChunkInfo]) -> float:
        """Compute keyword overlap between query and chunks."""
        
        # Extract keywords from query
        stopwords = {"của", "và", "là", "được", "có", "trong", "cho", "với", "theo",
                     "các", "những", "này", "đó", "để", "về", "từ", "tại", "khi",
                     "nào", "gì", "như", "thế", "sao", "ai", "đâu", "bao", "thì"}
        
        query_words = set(w.lower() for w in re.findall(r'\b\w+\b', query)
                         if len(w) > 1 and w.lower() not in stopwords)
        
        if not query_words:
            return 0.0
        
        # Check coverage in chunks
        all_chunk_text = " ".join(c.text.lower() for c in chunks[:5])
        matched = sum(1 for w in query_words if w in all_chunk_text)
        
        return matched / len(query_words)
    
    def _generate_generic_clarification(self, query: str) -> str:
        """Generate generic clarification question."""
        
        # Analyze query to generate relevant question
        query_lower = query.lower()
        
        if "ai" in query_lower or "thẩm quyền" in query_lower:
            return "Bạn muốn hỏi về thẩm quyền của cơ quan/cấp nào cụ thể?"
        elif "như thế nào" in query_lower or "thủ tục" in query_lower:
            return "Bạn có thể cho biết cụ thể loại thủ tục/dịch vụ bạn muốn tìm hiểu?"
        elif "bao nhiêu" in query_lower or "mức" in query_lower:
            return "Bạn đang hỏi về mức phí/thời gian/số lượng cụ thể nào?"
        else:
            return "Bạn có thể cung cấp thêm thông tin chi tiết về câu hỏi?"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BUILD FALLBACK RESPONSES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def build_ask_back_response(
        self,
        decision: FallbackDecision,
        include_partial_info: bool = True,
        chunks: Optional[List[ChunkInfo]] = None
    ) -> RAGOutput:
        """Build ask-back response."""
        
        answer_parts = []
        
        if include_partial_info and chunks:
            # Include what we found
            answer_parts.append("Tôi cần thêm thông tin để trả lời chính xác.")
            
            if chunks:
                top_chunk = chunks[0]
                answer_parts.append(f"\nThông tin liên quan tìm thấy trong {top_chunk.van_ban}:")
                # Summarize top chunk briefly
                summary = top_chunk.text[:200] + "..." if len(top_chunk.text) > 200 else top_chunk.text
                answer_parts.append(f"\"...{summary}...\"")
        
        answer_parts.append(f"\n\n❓ {decision.clarification_question}")
        
        return RAGOutput(
            answer="\n".join(answer_parts),
            citations=[],
            decision=DecisionType.ASK_BACK,
            abstain=True,
            abstain_reason=AbstainReason.AMBIGUOUS_QUERY,
            reason_detail=decision.reason,
            clarification_question=decision.clarification_question
        )
    
    def build_cautious_response(
        self,
        decision: FallbackDecision,
        generated_answer: str,
        citations: List[Citation]
    ) -> RAGOutput:
        """Build cautious response with warning.
        
        Cải tiến: Giữ nguyên answer của LLM nếu có, chỉ thêm warning nhẹ.
        Không wrap bằng cautious_prefix nếu LLM đã trả lời đầy đủ.
        """
        
        if generated_answer and generated_answer.strip():
            # LLM đã trả lời → giữ nguyên, chỉ thêm warning ngắn
            cautious_answer = generated_answer.strip()
            if citations:
                # Có citation → không cần warning nặng
                cautious_answer += "\n\n⚠️ Lưu ý: Vui lòng kiểm tra lại với văn bản gốc."
            else:
                cautious_answer += f"\n\n{decision.cautious_warning or ''}"
        else:
            # LLM không trả lời được → dùng template
            cautious_answer = f"{decision.cautious_prefix or ''}\n\n{decision.cautious_warning or ''}"
        
        return RAGOutput(
            answer=cautious_answer,
            citations=citations,
            decision=DecisionType.CAUTIOUS,
            abstain=False,
            reason_detail=decision.reason,
            supported_by_context=None  # Uncertain
        )
    
    def build_abstain_response(
        self,
        decision: FallbackDecision,
        query: str
    ) -> RAGOutput:
        """Build abstain response."""
        
        answer = self.cautious_templates["no_info"]
        
        # Add suggestion
        suggestions = [
            "Bạn có thể thử hỏi cách khác hoặc cung cấp thêm ngữ cảnh.",
            "Vui lòng tham vấn chuyên gia pháp lý để được tư vấn chính xác.",
        ]
        
        answer += f"\n\n💡 Gợi ý: {suggestions[0]}"
        
        return RAGOutput(
            answer=answer,
            citations=[],
            decision=DecisionType.ABSTAIN,
            abstain=True,
            abstain_reason=AbstainReason.NO_RELEVANT_CHUNK,
            reason_detail=decision.reason
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_fallback_if_needed(
    query: str,
    chunks: List[ChunkInfo],
    llm_response: Optional[RAGOutput] = None
) -> Tuple[RAGOutput, bool]:
    """
    Apply fallback strategy if needed.
    
    Args:
        query: User query
        chunks: Reranked chunks
        llm_response: Response from LLM (if generated)
    
    Returns:
        (final_response, was_fallback_applied)
    """
    
    strategy = FallbackStrategy()
    decision = strategy.evaluate(query, chunks)
    
    if decision.fallback_type == FallbackType.NONE:
        return llm_response, False
    
    if decision.fallback_type == FallbackType.ASK_BACK:
        response = strategy.build_ask_back_response(decision, chunks=chunks)
        return response, True
    
    if decision.fallback_type == FallbackType.CAUTIOUS:
        if llm_response:
            response = strategy.build_cautious_response(
                decision,
                llm_response.answer or "",
                llm_response.citations
            )
            return response, True
        else:
            # Generate cautious summary
            response = strategy.build_abstain_response(decision, query)
            return response, True
    
    if decision.fallback_type == FallbackType.ABSTAIN:
        response = strategy.build_abstain_response(decision, query)
        return response, True
    
    return llm_response, False


def get_cautious_warning() -> str:
    """Get standard cautious warning."""
    return "⚠️ **Lưu ý**: Thông tin trên chỉ mang tính tham khảo. Vui lòng tham vấn chuyên gia pháp lý để được tư vấn chính xác."
