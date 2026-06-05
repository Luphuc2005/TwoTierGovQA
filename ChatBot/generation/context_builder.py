"""
context_builder.py — Context Builder với Dedup, Span Trim, Metadata Injection

3 cải tiến chính để tăng Citation Hit Rate:
(a) Dedup & Diversify: Max 2 chunks/nguồn, ưu tiên chunks khác điều/khoản
(b) Span Trim: Cắt chunk chỉ còn đoạn liên quan
(c) Metadata Injection: Inject header pháp lý vào mỗi chunk
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from .rag_contract import ChunkInfo, RAGInput, RAGPolicy


class ContextBuilder:
    """
    Build context tối ưu cho LLM từ chunks đã rerank.
    
    Pipeline:
    1. Dedup & Diversify chunks
    2. Trim spans để giữ phần liên quan
    3. Inject metadata headers
    4. Build final context string
    """
    
    def __init__(self, policy: Optional[RAGPolicy] = None):
        self.policy = policy or RAGPolicy()
        
        # Legal keywords để detect query type
        self.legal_keywords = {
            "định nghĩa": ["định nghĩa", "là gì", "được hiểu", "khái niệm"],
            "điều kiện": ["điều kiện", "yêu cầu", "tiêu chuẩn", "đáp ứng"],
            "thủ tục": ["thủ tục", "trình tự", "hồ sơ", "quy trình"],
            "thẩm quyền": ["thẩm quyền", "có quyền", "chịu trách nhiệm", "thuộc"],
            "xử phạt": ["xử phạt", "phạt tiền", "vi phạm", "chế tài"],
            "phí": ["phí", "lệ phí", "chi phí", "thu phí"],
            "thời hạn": ["thời hạn", "trong vòng", "ngày", "tháng", "năm"]
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # (a) DEDUP & DIVERSIFY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def dedup_and_diversify(
        self, 
        chunks: List[ChunkInfo],
        max_per_source: int = 2,
        max_total: int = 5
    ) -> List[ChunkInfo]:
        """
        Loại bỏ chunks trùng lặp và đa dạng hóa nguồn.
        
        Rules:
        - Max 2 chunks từ cùng 1 văn bản + điều
        - Ưu tiên chunks từ các điều/khoản khác nhau
        - Giữ thứ tự rerank score
        """
        
        # Track: (van_ban, dieu) -> count
        source_count = defaultdict(int)
        
        # Track: exact text hash -> seen
        seen_texts = set()
        
        result = []
        
        for chunk in chunks:
            if len(result) >= max_total:
                break
            
            # Tạo source key
            source_key = (chunk.van_ban, chunk.dieu)
            
            # Check duplicate text (fuzzy)
            text_hash = self._normalize_text_hash(chunk.text)
            if text_hash in seen_texts:
                continue
            
            # Check per-source limit
            if source_count[source_key] >= max_per_source:
                continue
            
            # Accept chunk
            seen_texts.add(text_hash)
            source_count[source_key] += 1
            result.append(chunk)
        
        return result
    
    def _normalize_text_hash(self, text: str) -> str:
        """Normalize text để detect duplicates."""
        # Remove whitespace, lowercase, keep first 200 chars
        normalized = re.sub(r'\s+', '', text.lower())[:200]
        return normalized
    
    # ═══════════════════════════════════════════════════════════════════════════
    # (b) SPAN TRIM
    # ═══════════════════════════════════════════════════════════════════════════
    
    def trim_span(
        self,
        chunk: ChunkInfo,
        query: str,
        context_window: int = 250,
        min_overlap_ratio: float = 0.1
    ) -> ChunkInfo:
        """
        Cắt chunk giữ lại phần liên quan đến query.
        
        Strategy:
        1. Tìm keyword matches (compound + single) trong chunk
        2. Expand mỗi match tới ranh giới câu (sentence boundary)
        3. Merge overlapping spans
        4. Nếu chunk chứa legal header (Điều/Khoản/Điểm) → giữ cả block
        5. Nếu không tìm thấy matches → giữ nguyên
        """
        
        text = chunk.text
        
        # Chunk ngắn → giữ nguyên (tránh cắt mất thông tin)
        if len(text) <= 400:
            chunk.trimmed_text = text
            return chunk
        
        query_keywords = self._extract_keywords(query)
        
        if not query_keywords:
            chunk.trimmed_text = text
            return chunk
        
        # Tìm tất cả keyword positions (compound trước, single sau)
        spans = []
        text_lower = text.lower()
        
        for keyword in query_keywords:
            pattern = re.escape(keyword)
            for match in re.finditer(pattern, text_lower):
                # Expand tới sentence boundary thay vì fixed chars
                start = self._snap_to_sentence_start(text, match.start(), context_window)
                end = self._snap_to_sentence_end(text, match.end(), context_window)
                spans.append((start, end))
        
        # Nếu chunk chứa legal structure (Điều X, Khoản Y) → giữ toàn bộ block
        legal_block = self._find_legal_block(text, spans)
        if legal_block:
            spans.append(legal_block)
        
        if not spans:
            chunk.trimmed_text = text
            return chunk
        
        # Merge overlapping spans
        merged_spans = self._merge_spans(spans)
        
        # Extract và join trimmed text
        trimmed_parts = []
        for start, end in merged_spans:
            part = text[start:end]
            if start > 0:
                part = "..." + part.lstrip()
            if end < len(text):
                part = part.rstrip() + "..."
            trimmed_parts.append(part)
        
        chunk.trimmed_text = " ".join(trimmed_parts)
        chunk.highlight_spans = merged_spans
        
        return chunk
    
    def _snap_to_sentence_start(self, text: str, pos: int, max_expand: int) -> int:
        """Mở rộng về trước tới đầu câu gần nhất (. hoặc \\n)."""
        search_start = max(0, pos - max_expand)
        region = text[search_start:pos]
        # Tìm dấu kết câu gần nhất phía trước
        for delim in ['\n', '. ', '.\n']:
            idx = region.rfind(delim)
            if idx != -1:
                return search_start + idx + len(delim)
        return search_start
    
    def _snap_to_sentence_end(self, text: str, pos: int, max_expand: int) -> int:
        """Mở rộng về sau tới cuối câu gần nhất."""
        search_end = min(len(text), pos + max_expand)
        region = text[pos:search_end]
        for delim in ['.\n', '. ', '\n']:
            idx = region.find(delim)
            if idx != -1:
                return pos + idx + len(delim)
        return search_end
    
    def _find_legal_block(self, text: str, keyword_spans: list) -> tuple:
        """
        Nếu keyword nằm trong 1 block Điều/Khoản → trả về span bao trùm block đó.
        Giữ nguyên cả đơn vị pháp lý thay vì cắt giữa chừng.
        """
        if not keyword_spans:
            return None
        # Tìm tất cả header pháp lý
        headers = list(re.finditer(
            r'(?:^|\n)\s*(?:Điều|Khoản|Điểm)\s+\d+',
            text, re.MULTILINE
        ))
        if not headers:
            return None
        
        # Với mỗi keyword span, tìm block chứa nó
        first_kw_start = keyword_spans[0][0]
        last_kw_end = keyword_spans[-1][1]
        
        block_start = 0
        block_end = len(text)
        
        for i, h in enumerate(headers):
            if h.start() <= first_kw_start:
                block_start = h.start()
            if h.start() > last_kw_end:
                block_end = h.start()
                break
        
        # Chỉ trả về nếu block không quá dài (tránh giữ nguyên chunk 2000+ chars)
        if block_end - block_start <= 800:
            return (block_start, block_end)
        return None
    
    # Vietnamese stopwords — bao gồm cả từ pháp lý phổ biến không mang nghĩa phân biệt
    _VN_STOPWORDS = {
        "của", "và", "là", "được", "có", "trong", "cho", "với", "theo",
        "các", "những", "này", "đó", "để", "về", "từ", "tại", "khi",
        "nào", "gì", "như", "thế", "sao", "ai", "đâu", "bao",
        "không", "hay", "hoặc", "mà", "thì", "cũng", "đã", "sẽ",
        "rất", "lại", "nếu", "vì", "do", "bởi", "một", "người",
        "trên", "dưới", "sau", "trước", "đến", "bị", "phải",
        "còn", "nên", "đều", "chỉ", "ra", "vào", "lên",
    }

    # Compound từ pháp lý phổ biến (2-3 từ) — match trước khi tách đơn lẻ
    _LEGAL_COMPOUNDS = [
        # 3-gram trước
        "ủy ban nhân dân", "hội đồng nhân dân", "bảo hiểm xã hội",
        "bảo hiểm y tế", "quyền sử dụng đất", "giấy chứng nhận",
        "cơ quan nhà nước", "trách nhiệm hình sự", "trách nhiệm dân sự",
        "thi hành án", "cấp giấy phép", "đăng ký kinh doanh",
        "an ninh quốc gia", "an toàn giao thông", "bảo vệ môi trường",
        "giải quyết tranh chấp", "quyền sở hữu", "sở hữu trí tuệ",
        # 2-gram
        "thẩm quyền", "xử phạt", "vi phạm", "nghị định", "thông tư",
        "quyết định", "văn bản", "pháp luật", "quy định", "điều kiện",
        "thủ tục", "hồ sơ", "đăng ký", "cấp phép", "giấy phép",
        "lệ phí", "thuế thu", "thu nhập", "doanh nghiệp", "tổ chức",
        "cá nhân", "hộ gia đình", "thời hạn", "thời hiệu",
        "hình sự", "dân sự", "hành chính", "lao động", "bồi thường",
        "khiếu nại", "tố cáo", "xử lý", "trách nhiệm",
        "chế tài", "biện pháp", "quy hoạch", "xây dựng",
        "đất đai", "nhà ở", "giao thông", "môi trường",
        "sở hữu", "chuyển nhượng", "thừa kế", "hợp đồng",
        "bảo hiểm", "ngân sách", "tài chính", "kinh doanh",
        "đầu tư", "xuất khẩu", "nhập khẩu", "hải quan",
    ]

    def _extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """
        Extract keywords từ text, hỗ trợ tiếng Việt compound words.
        
        Strategy:
        1. Match compound pháp lý trước (2-3 từ: "thẩm quyền", "ủy ban nhân dân")
        2. Sau đó tách single tokens cho phần còn lại
        3. Ưu tiên compound → single (compound match trước khi search)
        """
        text_lower = text.lower()
        keywords = []
        seen = set()
        
        # Phase 1: Match compound keywords (ưu tiên dài trước)
        remaining = text_lower
        for compound in self._LEGAL_COMPOUNDS:
            if compound in text_lower and compound not in seen:
                seen.add(compound)
                keywords.append(compound)
        
        # Phase 2: Single-word tokens (bỏ stopwords, bỏ từ đã nằm trong compound)
        words = re.findall(r'[\w]+', text_lower)
        compound_words = set()
        for c in keywords:
            compound_words.update(c.split())
        
        for w in words:
            if (len(w) >= min_length
                    and w not in self._VN_STOPWORDS
                    and w not in seen
                    and w not in compound_words):
                seen.add(w)
                keywords.append(w)
        
        return keywords
    
    def _merge_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping spans."""
        if not spans:
            return []
        
        sorted_spans = sorted(spans, key=lambda x: x[0])
        merged = [sorted_spans[0]]
        
        for start, end in sorted_spans[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                # Overlap → merge
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        
        return merged
    
    # ═══════════════════════════════════════════════════════════════════════════
    # (c) METADATA INJECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def inject_metadata(self, chunk: ChunkInfo) -> str:
        """
        Inject metadata header vào chunk text.
        
        Format:
        [VB: Nghị định..., Chương II, Điều 5, Khoản 2, Điểm a]
        <text>
        """
        header = chunk.get_metadata_header()
        text = chunk.trimmed_text or chunk.text
        return f"{header}\n{text}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN BUILD PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def build(
        self,
        chunks: List[ChunkInfo],
        query: str,
        max_chunks: int = 5,
        enable_dedup: bool = True,
        enable_trim: bool = True,
        enable_metadata: bool = True
    ) -> Tuple[List[ChunkInfo], str]:
        """
        Build context từ chunks.
        
        Returns:
            (processed_chunks, context_string)
        """
        
        processed = list(chunks)
        
        # Step 1: Dedup & Diversify
        if enable_dedup:
            processed = self.dedup_and_diversify(
                processed,
                max_per_source=self.policy.max_chunks_per_source,
                max_total=max_chunks
            )
        else:
            processed = processed[:max_chunks]
        
        # Step 2: Trim spans
        if enable_trim:
            processed = [self.trim_span(chunk, query) for chunk in processed]
        
        # Step 3: Build context string
        context_parts = []
        for i, chunk in enumerate(processed, 1):
            if enable_metadata:
                text_with_meta = self.inject_metadata(chunk)
            else:
                text_with_meta = chunk.trimmed_text or chunk.text
            
            context_parts.append(f"[{i}] {text_with_meta}")
        
        context_string = "\n\n".join(context_parts)
        
        return processed, context_string
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_query_type(self, query: str) -> List[str]:
        """Detect loại query dựa trên keywords."""
        query_lower = query.lower()
        detected_types = []
        
        for qtype, keywords in self.legal_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_types.append(qtype)
        
        return detected_types
    
    def check_keyword_coverage(
        self,
        query: str,
        chunks: List[ChunkInfo]
    ) -> Dict[str, any]:
        """
        Kiểm tra chunks có chứa keywords từ query không.
        Dùng cho gating decision.
        """
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return {"coverage": 0.0, "missing": [], "found": []}
        
        # Check each keyword
        found = []
        missing = []
        
        all_chunk_text = " ".join(c.text.lower() for c in chunks)
        
        for kw in query_keywords:
            if kw in all_chunk_text:
                found.append(kw)
            else:
                missing.append(kw)
        
        coverage = len(found) / len(query_keywords) if query_keywords else 0.0
        
        return {
            "coverage": round(coverage, 4),
            "found": found,
            "missing": missing,
            "total_keywords": len(query_keywords)
        }
    
    def needs_clarification(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Kiểm tra query có cần clarification không.
        
        Returns:
            (needs_clarification, clarification_question)
        """
        query_lower = query.lower()
        
        # Check ambiguous patterns
        ambiguous_patterns = [
            (r'\bnày\b(?! (?:là|quy|nghị))', "Bạn đang hỏi về văn bản nào cụ thể?"),
            (r'\bđó\b', "Bạn có thể cho biết cụ thể hơn về đối tượng/văn bản?"),
            (r'^(?:ai|gì|sao|nào)\??$', "Câu hỏi hơi ngắn, bạn có thể mô tả chi tiết hơn?"),
        ]
        
        for pattern, question in ambiguous_patterns:
            if re.search(pattern, query_lower):
                return True, question
        
        # Check missing context
        missing_context_triggers = [
            ("lĩnh vực", "Bạn đang hỏi về lĩnh vực quản lý nào?"),
            ("địa phương", "Đây là địa phương/tỉnh/thành phố nào?"),
            ("đối tượng", "Đối tượng áp dụng cụ thể là ai?"),
        ]
        
        for trigger, question in missing_context_triggers:
            if trigger in query_lower and len(query.split()) < 10:
                return True, question
        
        return False, None


class ContextOptimizer:
    """
    Tối ưu context để tăng Citation Hit Rate và giảm hallucination.
    """
    
    def __init__(self, context_builder: Optional[ContextBuilder] = None):
        self.builder = context_builder or ContextBuilder()
    
    def optimize_for_legal_qa(
        self,
        rag_input: RAGInput,
        aggressive_trim: bool = False
    ) -> RAGInput:
        """
        Optimize RAGInput cho Legal QA.
        
        Steps:
        1. Dedup & diversify chunks
        2. Trim spans
        3. Reorder by relevance
        """
        
        chunks = rag_input.top_k_chunks
        query = rag_input.question
        policy = rag_input.policy
        
        # Build optimized context
        processed_chunks, context_string = self.builder.build(
            chunks=chunks,
            query=query,
            max_chunks=policy.max_chunks_in_context,
            enable_dedup=True,
            enable_trim=True,
            enable_metadata=True
        )
        
        # Update RAGInput
        rag_input.top_k_chunks = processed_chunks
        
        return rag_input
    
    def estimate_context_tokens(self, context: str) -> int:
        """Ước tính số tokens (rough estimate for Vietnamese)."""
        # Vietnamese: ~1.5 chars per token average
        return int(len(context) / 1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_context_for_generation(
    chunks: List[Dict],
    query: str,
    policy: Optional[RAGPolicy] = None,
    enable_optimization: bool = True
) -> Tuple[List[ChunkInfo], str]:
    """
    Convenience function để build context từ raw chunks.
    
    Args:
        chunks: List of chunk dicts từ retriever
        query: User query
        policy: RAG policy
        enable_optimization: Bật/tắt optimization
    
    Returns:
        (chunk_infos, context_string)
    """
    
    policy = policy or RAGPolicy()
    
    # Convert to ChunkInfo
    chunk_infos = []
    for chunk in chunks:
        meta = chunk.get("metadata", chunk.get("meta", {}))
        info = ChunkInfo(
            chunk_id=chunk.get("chunk_id", chunk.get("faiss_id", -1)),
            text=chunk.get("text", chunk.get("passage", "")),
            score_retrieval=chunk.get("score_retrieval", 0.0),
            score_rerank=chunk.get("score_rerank", 0.0),
            van_ban=meta.get("van_ban", ""),
            chuong=meta.get("chuong"),
            dieu=meta.get("dieu"),
            khoan=meta.get("khoan"),
            diem=meta.get("diem"),
            source_file=meta.get("source_file", ""),
        )
        chunk_infos.append(info)
    
    builder = ContextBuilder(policy)
    
    if enable_optimization:
        return builder.build(
            chunks=chunk_infos,
            query=query,
            max_chunks=policy.max_chunks_in_context,
            enable_dedup=True,
            enable_trim=True,
            enable_metadata=True
        )
    else:
        # No optimization, just format
        context_parts = []
        for i, chunk in enumerate(chunk_infos[:policy.max_chunks_in_context], 1):
            context_parts.append(f"[{i}] {chunk.text}")
        return chunk_infos[:policy.max_chunks_in_context], "\n\n".join(context_parts)
