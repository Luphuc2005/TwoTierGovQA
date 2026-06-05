"""
rag_contract.py — Định nghĩa Input/Output Contract cho Legal RAG System

Mục tiêu:
- Chuẩn hóa format đầu vào/đầu ra cho LLM
- Dễ dàng đo lường: Answer Pass Rate, Faithfulness, Citation Correctness
- Tương thích với các LLM khác nhau (local llama.cpp, API, etc.)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import re


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class LLMTier(Enum):
    """Tầng LLM để xử lý query."""
    LOCAL = "local"             # Tầng 1: Local model, nhanh, rẻ
    API = "api"                 # Tầng 2: API (Gemini/GPT), chính xác hơn
    NONE = "none"               # Không cần LLM (abstain/ask_back)


class DecisionType(Enum):
    ANSWER = "answer"           # Trả lời đầy đủ với citation
    ABSTAIN = "abstain"         # Không đủ căn cứ để trả lời
    ASK_BACK = "ask_back"       # Cần thêm thông tin từ user
    CAUTIOUS = "cautious"       # Trả lời thận trọng với cảnh báo


class AbstainReason(Enum):
    NO_RELEVANT_CHUNK = "no_relevant_chunk"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_LEGAL_KEYWORD = "missing_legal_keyword"
    AMBIGUOUS_QUERY = "ambiguous_query"
    OUT_OF_SCOPE = "out_of_scope"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INPUT CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkInfo:
    """Thông tin một chunk được retrieve/rerank."""
    
    chunk_id: int                      # ID trong FAISS index
    text: str                          # Nội dung chunk
    score_retrieval: float             # Score từ bi-encoder/FAISS
    score_rerank: float                # Score từ cross-encoder
    
    # Metadata pháp lý
    van_ban: str = ""                  # Tên văn bản (Nghị định, Thông tư, ...)
    chuong: Optional[str] = None       # Chương
    dieu: Optional[str] = None         # Điều
    khoan: Optional[str] = None        # Khoản
    diem: Optional[str] = None         # Điểm
    source_file: str = ""              # File nguồn
    page_number: Optional[int] = None   # Số trang
    
    # Span info (sau khi trim)
    trimmed_text: Optional[str] = None # Text đã được trim
    highlight_spans: List[tuple] = field(default_factory=list)  # [(start, end), ...]
    
    def get_citation_str(self) -> str:
        """Tạo chuỗi citation chuẩn: Điều X, Khoản Y, Điểm Z."""
        parts = []
        if self.dieu:
            parts.append(f"Điều {self.dieu}")
        if self.khoan:
            parts.append(f"Khoản {self.khoan}")
        if self.diem:
            parts.append(f"Điểm {self.diem}")
        return ", ".join(parts) if parts else "N/A"
    
    def get_full_citation(self) -> str:
        """Citation đầy đủ với tên văn bản."""
        citation = self.get_citation_str()
        if self.van_ban:
            return f"{citation} - {self.van_ban}"
        return citation
    
    def get_metadata_header(self) -> str:
        """Tạo header metadata để inject vào context."""
        parts = [f"[VB: {self.van_ban}"]
        if self.chuong:
            parts.append(f"Chương {self.chuong}")
        if self.dieu:
            parts.append(f"Điều {self.dieu}")
        if self.khoan:
            parts.append(f"Khoản {self.khoan}")
        if self.diem:
            parts.append(f"Điểm {self.diem}")
        return ", ".join(parts) + "]"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGPolicy:
    """Chính sách bắt buộc cho Legal RAG."""
    
    # Luật cứng
    must_cite: bool = True              # Bắt buộc trích dẫn điều/khoản
    no_hallucination: bool = True       # Không được bịa thông tin
    abstain_if_uncertain: bool = True   # Abstain nếu không đủ căn cứ
    
    # Ngưỡng quyết định
    confidence_threshold_pass: float = 0.5    # Score tối thiểu để PASS
    confidence_threshold_abstain: float = 0.3 # Dưới mức này → ABSTAIN
    margin_threshold: float = 0.2             # Khoảng cách top1-top2
    
    # Giới hạn
    max_chunks_in_context: int = 5
    max_chunks_per_source: int = 2      # Dedup: tối đa 2 chunk/văn bản
    min_keyword_overlap: float = 0.3    # Overlap từ khóa tối thiểu
    
    # Ask-back triggers
    ask_back_keywords: List[str] = field(default_factory=lambda: [
        "địa phương", "thời điểm", "đối tượng", "trường hợp", "cụ thể"
    ])


@dataclass
class RAGInput:
    """Input chuẩn cho LLM trong Legal RAG."""
    
    question: str                       # Câu hỏi từ user
    top_k_chunks: List[ChunkInfo]       # Danh sách chunks đã rerank
    policy: RAGPolicy                   # Chính sách RAG
    
    # Metadata bổ sung
    query_id: Optional[str] = None
    retrieved_at: Optional[str] = None
    total_retrieve_time_ms: float = 0.0
    total_rerank_time_ms: float = 0.0
    
    # Gating decision (từ Gating module)
    gating_decision: Optional[DecisionType] = None
    gating_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["gating_decision"] = self.gating_decision.value if self.gating_decision else None
        return result
    
    def get_context_for_llm(self) -> str:
        """Tạo context string để đưa vào prompt."""
        context_parts = []
        for i, chunk in enumerate(self.top_k_chunks, 1):
            header = chunk.get_metadata_header()
            text = chunk.trimmed_text or chunk.text
            context_parts.append(f"[{i}] {header}\n{text}")
        return "\n\n".join(context_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OUTPUT CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Citation:
    """Một trích dẫn pháp lý trong câu trả lời."""
    
    van_ban: str                        # Tên văn bản
    chuong: Optional[str] = None
    dieu: Optional[str] = None
    khoan: Optional[str] = None
    diem: Optional[str] = None
    
    # Liên kết với chunk
    chunk_id: Optional[int] = None      # ID chunk nguồn
    span_start: Optional[int] = None    # Vị trí bắt đầu trong text
    span_end: Optional[int] = None      # Vị trí kết thúc trong text
    quoted_text: Optional[str] = None   # Đoạn text được trích dẫn
    page_number: Optional[int] = None   # Số trang trong PDF
    source_file: Optional[str] = None   # Đường dẫn file PDF
    
    def matches(self, expected: Dict) -> bool:
        """Kiểm tra citation có khớp với expected không."""
        # Phải có ít nhất 1 field để match — không match nếu cả 2 bên đều rỗng
        has_any_field = False
        
        if self.dieu and expected.get("dieu"):
            has_any_field = True
            if str(self.dieu).strip() != str(expected.get("dieu")).strip():
                return False
        elif self.dieu or expected.get("dieu"):
            # Một bên có dieu, bên kia không → không match
            return False
            
        if self.khoan and expected.get("khoan"):
            has_any_field = True
            if str(self.khoan).strip() != str(expected.get("khoan")).strip():
                return False
        
        if self.diem and expected.get("diem"):
            has_any_field = True
            if str(self.diem).strip() != str(expected.get("diem")).strip():
                return False
        
        # Phải có ít nhất 1 trường được so khớp
        return has_any_field
    
    def to_str(self) -> str:
        parts = []
        if self.dieu:
            parts.append(f"Điều {self.dieu}")
        if self.khoan:
            parts.append(f"Khoản {self.khoan}")
        if self.diem:
            parts.append(f"Điểm {self.diem}")
        cite_loc = ", ".join(parts) if parts else "N/A"
        if self.van_ban and self.van_ban != "N/A":
            return f"**{self.van_ban}**: {cite_loc}"
        return cite_loc
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGOutput:
    """Output chuẩn từ LLM trong Legal RAG."""
    
    # Kết quả chính
    answer: Optional[str] = None        # Câu trả lời tiếng Việt
    citations: List[Citation] = field(default_factory=list)
    
    # Quyết định
    decision: DecisionType = DecisionType.ANSWER
    abstain: bool = False
    abstain_reason: Optional[AbstainReason] = None
    reason_detail: Optional[str] = None  # Chi tiết lý do abstain
    
    # Ask-back (nếu cần)
    clarification_question: Optional[str] = None
    
    # Metadata
    confidence_score: float = 0.0       # Điểm tin cậy tổng
    latency_llm_ms: float = 0.0         # Thời gian LLM inference
    raw_response: Optional[str] = None  # Response gốc từ LLM
    
    # Evaluation markers
    supported_by_context: Optional[bool] = None  # Answer có được support bởi context?
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["decision"] = self.decision.value
        result["abstain_reason"] = self.abstain_reason.value if self.abstain_reason else None
        result["citations"] = [c.to_dict() for c in self.citations]
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FACTORY & PARSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_rag_input(
    question: str,
    chunks: List[Dict],
    policy: Optional[RAGPolicy] = None
) -> RAGInput:
    """Tạo RAGInput từ kết quả retrieve/rerank."""
    
    if policy is None:
        policy = RAGPolicy()
    
    chunk_infos = []
    for chunk in chunks:
        meta = chunk.get("metadata", chunk.get("meta", {}))
        chunk_info = ChunkInfo(
            chunk_id=chunk.get("chunk_id", chunk.get("faiss_id", -1)),
            text=chunk.get("text", chunk.get("passage", "")),
            score_retrieval=chunk.get("score_retrieval", chunk.get("faiss_score", 0.0)),
            score_rerank=chunk.get("score_rerank", chunk.get("rerank_score", 0.0)),
            van_ban=meta.get("van_ban", ""),
            chuong=meta.get("chuong"),
            dieu=meta.get("dieu"),
            khoan=meta.get("khoan"),
            diem=meta.get("diem"),
            source_file=meta.get("source_file", ""),
        )
        chunk_infos.append(chunk_info)
    
    return RAGInput(
        question=question,
        top_k_chunks=chunk_infos,
        policy=policy
    )


def parse_rag_output(raw_response: str, expected_format: str = "structured") -> RAGOutput:
    """
    Parse response từ LLM thành RAGOutput.
    
    Hỗ trợ nhiều format:
    1. JSON trong markdown code block (```json ... ```)
    2. Raw JSON object
    3. Natural text với citation markers
    """
    
    output = RAGOutput(raw_response=raw_response)
    
    if not raw_response or not raw_response.strip():
        output.abstain = True
        output.decision = DecisionType.ABSTAIN
        output.reason_detail = "Empty response from LLM"
        return output
    
    # ── Thử parse JSON ──
    json_text = None
    
    # Ưu tiên 1: JSON trong markdown code block (```json ... ``` hoặc ``` ... ```)
    md_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)```', raw_response)
    if md_match:
        json_text = md_match.group(1).strip()
    
    # Ưu tiên 2: Raw JSON object
    if not json_text:
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            json_text = json_match.group()
    
    # Ưu tiên 3: Trường hợp đặc biệt: Thiếu dấu { ở đầu (thường gặp khi LLM output lỗi)
    # Tìm pattern "key": ... }
    if not json_text:
        broken_json_match = re.search(r'"[a-zA-Z_]+"\s*:(?:.|\n)*\}', raw_response, re.DOTALL)
        if broken_json_match:
            json_text = broken_json_match.group()
            if not json_text.startswith('{'):
                json_text = '{' + json_text
    
    if json_text:
        # Fallback: check if the text contains something that looks like a JSON object but is missing braces
        if not json_text.strip().startswith('{'):
            # Try to wrap in braces if it looks like key-value pairs
            potential_json = '{' + json_text.strip() + '}'
            try:
                data = json.loads(potential_json)
                json_text = potential_json
            except:
                pass

        try:
            data = json.loads(json_text)
            
            # Extract fields
            if isinstance(data, dict):
                # Check for abstain or refusal
                is_abstain = data.get("abstain", False) or "refusal" in data
                
                if is_abstain:
                    output.abstain = True
                    output.decision = DecisionType.ABSTAIN
                    output.reason_detail = (
                        data.get("reason") or 
                        data.get("ly_do") or 
                        data.get("refusal") or 
                        ""
                    )
                    output.answer = data.get("answer") or output.reason_detail
                else:
                    output.answer = (
                        data.get("answer") or 
                        data.get("ket_luan") or 
                        data.get("tra_loi") or 
                        data.get("response")
                    )
                
                # Citations
                citations_data = data.get("citations") or data.get("trich_dan") or []
                if not isinstance(citations_data, list) and isinstance(citations_data, dict):
                    # Trường hợp đặc biệt: citations là 1 dict thay vì list
                    citations_data = [citations_data]
                    
                for cit in citations_data:
                    if isinstance(cit, dict):
                        # Kiểm tra xem có bất kỳ field nào có giá trị không
                        has_val = any([
                            cit.get("dieu"), cit.get("khoan"), 
                            cit.get("diem"), cit.get("van_ban")
                        ])
                        if has_val:
                            citation = Citation(
                                van_ban=cit.get("van_ban", ""),
                                dieu=str(cit.get("dieu", "")) if cit.get("dieu") else None,
                                khoan=str(cit.get("khoan", "")) if cit.get("khoan") else None,
                                diem=str(cit.get("diem", "")) if cit.get("diem") else None,
                                quoted_text=cit.get("quoted_text") or cit.get("noi_dung")
                            )
                            output.citations.append(citation)
            
            # If answer is empty but we have JSON, maybe it's just a string or in the raw response
            if not output.answer and not output.abstain:
                if isinstance(data, str):
                    output.answer = data
            
            # CHỈ return nếu đã tìm thấy cả answer và citations, hoặc là abstain.
            # Nếu có answer nhưng thiếu citations, cho phép chạy tiếp để tìm citations bằng Regex.
            if (output.answer and output.citations) or output.abstain:
                return output
            
        except (json.JSONDecodeError, AttributeError, KeyError):
            pass  # Fall through to natural text parsing
    
    # ── Fallback: parse natural text (Regex) ──
    # Tại đây có 2 trường hợp:
    # 1. JSON parse lỗi/không có → output.answer đang rỗng
    # 2. JSON parse ok nhưng citations rỗng → output.answer đã có data
    
    source_for_regex = output.answer if output.answer else raw_response
    clean_text = source_for_regex.strip()
    
    # Remove junk prefix like ": " or ': "' which sometimes appears from Gemini JSON completion
    clean_text = re.sub(r'^[:\s"]+', '', clean_text)
    
    # Nếu chưa có answer từ JSON, tìm "Kết luận" hoặc dùng toàn bộ text
    if not output.answer:
        conclusion_match = re.search(
            r'(?:\*\*)?(?:Kết luận|Câu trả lời|Trả lời|Answer|Kết quả)(?:\*\*)?[:\s\-]*(.+?)(?:\n\n|\n\*\*|\nTrích dẫn|$)',
            clean_text, re.IGNORECASE | re.DOTALL
        )
        if conclusion_match:
            output.answer = conclusion_match.group(1).strip()
        else:
            # Nếu không có "Kết luận", nhưng có "Trích dẫn" ở cuối, lấy phần trước đó
            cite_header_match = re.search(r'\n\s*(?:Trích dẫn|Căn cứ pháp lý|📎 \*\*Trích dẫn:\*\*)\s*\n', clean_text, re.IGNORECASE)
            if cite_header_match:
                output.answer = clean_text[:cite_header_match.start()].strip()
            else:
                output.answer = clean_text

    # ── Tìm các pattern citation trong text (Robust parsing) ──
    # Luôn tìm trong raw_response để đảm bảo không sót citation nằm ngoài answer block
    search_text = raw_response
    
    # Regex cho Điều X, Khoản Y, Điểm Z
    dieu_pattern = r'(?:Điều|điều)(?:\s+số)?\s*(\d+[a-z]?)'
    khoan_pattern = r'(?:Khoản|khoản)(?:\s+số)?\s*(\d+)'
    diem_pattern = r'(?:Điểm|điểm)(?:\s+số)?\s*([a-zđ])'
    
    # 1. Tìm toàn bộ các Điều
    dieu_matches = list(re.finditer(dieu_pattern, search_text))
    
    for d_match in dieu_matches:
        dieu_val = d_match.group(1)
        
        # Tìm Khoản/Điểm gần đó: check cả TRƯỚC (50 chars) và SAU (100 chars)
        start_search = max(0, d_match.start() - 60)
        end_search = min(len(search_text), d_match.end() + 100)
        context_near = search_text[start_search:end_search]
        
        k_match = re.search(khoan_pattern, context_near)
        khoan_val = k_match.group(1) if k_match else None
        
        di_match = re.search(diem_pattern, context_near)
        diem_val = di_match.group(1) if di_match else None
        
        # Tránh duplicate
        is_dup = any(c.dieu == dieu_val and c.khoan == khoan_val and c.diem == diem_val for c in output.citations)
        if not is_dup:
            output.citations.append(Citation(van_ban="", dieu=dieu_val, khoan=khoan_val, diem=diem_val))

    # Fallback nếu không có bất kỳ citation nào (từ JSON hoặc Regex ở trên)
    if not output.citations:
        k_matches = re.finditer(khoan_pattern, raw_response)
        for k_match in k_matches:
            k_val = k_match.group(1)
            is_dup = any(c.khoan == k_val for c in output.citations)
            if not is_dup:
                output.citations.append(Citation(van_ban="", dieu=None, khoan=k_val))

    # Kiểm tra abstain markers (VN + EN) - Robust check
    abstain_markers = [
        # Vietnamese
        "không tìm thấy", "không đủ căn cứ", "không có thông tin",
        "thiếu thông tin", "không thể xác định", "cần thêm thông tin",
        "không có quy định", "không được đề cập", "không có dữ liệu",
        "không liên quan", "không cung cấp", "không thuộc phạm vi",
        # English
        "not provided", "not found", "no information", "cannot answer",
        "cannot determine", "no relevant", "not mentioned",
        "i cannot", "i am unable", "not available", "no specific provision",
    ]
    resp_lower = raw_response.lower().strip()
    
    # Phát hiện response chỉ là từ "Abstain" hoặc câu ngắn từ chối
    if resp_lower in ("abstain", "[abstain]", "abstain.", "tôi không biết.", "không tìm thấy."):
        output.answer = None
        output.abstain = True
        output.decision = DecisionType.ABSTAIN
        output.reason_detail = "LLM explicitly abstained"
        return output
    
    # Nếu response quá ngắn và chứa marker -> ABSTAIN
    if len(resp_lower) < 200 and any(marker in resp_lower for marker in abstain_markers):
        output.abstain = True
        output.decision = DecisionType.ABSTAIN
        if not output.reason_detail:
            output.reason_detail = clean_text # Dùng chính câu trả lời làm reason
    
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_output_against_policy(output: RAGOutput, policy: RAGPolicy) -> List[str]:
    """Kiểm tra output có tuân thủ policy không."""
    violations = []
    
    # Kiểm tra phải có citation
    if policy.must_cite and output.answer and not output.abstain:
        if len(output.citations) == 0:
            violations.append("VIOLATION: Answer without citation (must_cite=True)")
    
    return violations


def compute_citation_correctness(
    output: RAGOutput, 
    expected_citations: List[Dict]
) -> Dict[str, float]:
    """Tính toán Citation Correctness metrics."""
    
    if not expected_citations:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    
    if not output.citations:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}
    
    # Count matches
    matched_expected = set()
    matched_output = set()
    
    for i, out_cit in enumerate(output.citations):
        for j, exp_cit in enumerate(expected_citations):
            if out_cit.matches(exp_cit):
                matched_expected.add(j)
                matched_output.add(i)
    
    precision = len(matched_output) / len(output.citations) if output.citations else 0
    recall = len(matched_expected) / len(expected_citations) if expected_citations else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    exact_match = 1.0 if len(matched_expected) == len(expected_citations) else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": exact_match
    }
