import re
from typing import Dict, List, Union

class LawParser:
    def __init__(self):
        # ======================================================================
        # 1. REGEX CẤU TRÚC BODY (Thêm Tolerance cho lỗi OCR)
        # ======================================================================
        self.re_chuong = re.compile(r'^\s*(?:Chương|Chuong)\s+([IVXLCDM0-9]+)', re.IGNORECASE)
        self.re_muc = re.compile(r'^\s*(?:Mục|Muc)\s+(\d+)', re.IGNORECASE)
        self.re_dieu = re.compile(r'^\s*(?:Điều|Diu|Diều|Biều|Ðiều|Diéu|Dieu|Diêu)\s+(\d+)\.[\s\t]*(.*)$', re.IGNORECASE)
        self.re_khoan = re.compile(r'^\s*(\d{1,2})\.[\s\t]*(.*)$')
        self.re_diem = re.compile(r'^\s*([a-zA-ZđĐ])\)\s*(.*)$')
        self.re_diem_split = re.compile(r'(?:[:;.]\s*|^.{0,15}\s+)([a-zA-ZđĐ])\)\s+(.*)$')
        self.re_phu_luc_start = re.compile(r'^\s*(?:Phụ\s+lục|Phu\s+luc)\b', re.IGNORECASE)

        # ======================================================================
        # 2. REGEX HEADER
        # ======================================================================
        self.re_doc_type = re.compile(
            r'^\s*(LUẬT|BỘ LUẬT|NGHỊ ĐỊNH|NGHI DINH|THÔNG TƯ|THONG TU|QUYẾT ĐỊNH|QUYET DINH|NGHỊ QUYẾT|NGHI QUYET|CHỈ THỊ|PHÁP LỆNH)\b', 
            re.IGNORECASE
        )
        # Bỏ "Chương" khỏi stop words để tránh conflict logic
        self.re_header_stop = re.compile(r'^\s*(Căn cứ|Điều|Kính gửi)\b', re.IGNORECASE)

        # ======================================================================
        # 3. REGEX DỌN DẸP
        # ======================================================================
        self.re_stop = re.compile(r'^\s*(Nơi nhận:|TM\.|KT\.|PP\.)', re.IGNORECASE)
        self.re_trash_page = re.compile(r'^\s*\d+\s*$') 
        self.re_trash_meta = re.compile(
            r'^\s*(NGƯỜI KÝ|Email|Thời gian ký|Cơ quan:|Page|source|CỘNG HÒA|Độc lập|Hạnh phúc|Số:|Hà Nội,)', 
            re.IGNORECASE
        )
        self.re_page_marker = re.compile(r'^\[PAGE_(\d+)\]$', re.IGNORECASE)

    def parse(self, text: Union[str, List[str]]) -> Dict:
        if isinstance(text, str):
            lines = text.splitlines()
        else:
            full_text = "\n".join(text)
            lines = full_text.splitlines()

        cleaned_lines = [l.strip() for l in lines if l.strip()]
        return self._process_lines(cleaned_lines)

    def _is_valid_phu_luc_header(self, line: str) -> bool:
        if not self.re_phu_luc_start.match(line): return False
        if len(line.split()) > 15: return False
        lower_line = line.lower()
        if "kèm theo" in lower_line or "ban hành" in lower_line or "tại khoản" in lower_line:
            return False
        return True

    def _append_text(self, node, text):
        if not node or not text: return
        if isinstance(node, dict):
            if "noi_dung" in node:
                if isinstance(node["noi_dung"], list):
                    node["noi_dung"].append(text)
                else:
                    node["noi_dung"] = (node["noi_dung"] + " " + text).strip()

    def _process_lines(self, lines: List[str]) -> Dict:
        result = {
            "ten_van_ban": "",
            "chuong": [],
            "phu_luc": []
        }
        
        cur_chuong = None
        cur_dieu = None
        cur_khoan = None
        cur_diem = None
        cur_phu_luc = None
        last_node = None 
        in_phu_luc_section = False
        current_page = 1
        
        doc_name_parts = []
        finished_doc_header = False

        for line in lines:
            # --- 0. LỌC RÁC & MARKER TRANG ---
            m_page = self.re_page_marker.match(line)
            if m_page:
                current_page = int(m_page.group(1))
                continue

            if self.re_trash_page.match(line): continue
            
            # --- 1. HEADER ---
            if not finished_doc_header and not cur_chuong and not cur_dieu:
                if self.re_header_stop.match(line):
                    finished_doc_header = True
                    result["ten_van_ban"] = " ".join(doc_name_parts).strip()
                elif doc_name_parts or self.re_doc_type.match(line):
                    if not self.re_trash_meta.match(line):
                         doc_name_parts.append(line)
                    # Lưu ý: Không continue để vẫn check được Chương/Điều
            
            # --- 2. FOOTER ---
            if self.re_stop.match(line) and not in_phu_luc_section:
                cur_chuong = None; cur_dieu = None; last_node = None
                if not result["ten_van_ban"] and doc_name_parts:
                     result["ten_van_ban"] = " ".join(doc_name_parts).strip()
                continue

            # --- 3. PHỤ LỤC ---
            if self._is_valid_phu_luc_header(line):
                in_phu_luc_section = True
                cur_phu_luc = {"tieu_de": line, "noi_dung": []}
                result["phu_luc"].append(cur_phu_luc)
                last_node = cur_phu_luc
                cur_chuong = None; cur_dieu = None
                continue
            
            if in_phu_luc_section and cur_phu_luc:
                cur_phu_luc["noi_dung"].append(line)
                continue

            # ==============================================================
            # BODY
            # ==============================================================

            # --- CHƯƠNG ---
            m_chuong = self.re_chuong.match(line)
            if m_chuong:
                # [QUAN TRỌNG] Logic chống bắt nhầm trích dẫn (Anti-Citation)
                # Nếu dòng là: "Chương III của Luật Hộ tịch" -> m_chuong vẫn match "Chương III"
                # Ta cần kiểm tra phần đuôi
                
                # Lấy phần text sau số chương
                # Ví dụ: line = "Chương III của Luật..." -> match end tại sau chữ III
                raw_suffix = line[m_chuong.end():].strip().lower()
                
                # Các từ khóa cho thấy đây là trích dẫn, không phải tiêu đề
                citation_keywords = ["của", "tại", "theo", "trong", "luật", "nghị định"]
                
                is_citation = False
                # Nếu ngay sau số chương là các từ nối -> Khả năng cao là trích dẫn
                for kw in citation_keywords:
                    if raw_suffix.startswith(kw):
                        is_citation = True
                        break
                
                if is_citation:
                    # Nếu là trích dẫn, coi như text bình thường, nối vào node cũ
                    if last_node: self._append_text(last_node, line)
                    continue # Bỏ qua việc tạo chương mới

                # --- NẾU LÀ CHƯƠNG THẬT ---
                if not finished_doc_header:
                    finished_doc_header = True
                    if doc_name_parts and doc_name_parts[-1] == line: doc_name_parts.pop()
                    result["ten_van_ban"] = " ".join(doc_name_parts).strip()

                cur_chuong = {"chuong_so": m_chuong.group(1), "chuong_ten": "", "dieu": [], "page_number": current_page}
                result["chuong"].append(cur_chuong)
                cur_dieu = None; last_node = cur_chuong
                continue
            
            if cur_chuong and not cur_chuong["chuong_ten"] and line.isupper() and len(line) > 5:
                cur_chuong["chuong_ten"] = line
                continue

            # --- MỤC (Bỏ qua nhưng không làm gãy dòng) ---
            if self.re_muc.match(line):
                # Hiện tại ta chưa lưu cấu trúc Mục, nên có thể bỏ qua hoặc nối vào Chương
                # Ở đây ta chọn bỏ qua để không lẫn vào nội dung Điều
                continue

            # --- ĐIỀU ---
            m_dieu = self.re_dieu.match(line)
            if "Diu 1" in line:
                print(f"[DEBUG] Found 'Diu 1' in line: {repr(line)}")
                print(f"[DEBUG] Using Regex: {self.re_dieu.pattern}")
                print(f"[DEBUG] Regex match result: {m_dieu}")
            
            if m_dieu:
                if not finished_doc_header:
                    finished_doc_header = True
                    if doc_name_parts and doc_name_parts[-1] == line: doc_name_parts.pop()
                    result["ten_van_ban"] = " ".join(doc_name_parts).strip()

                if not cur_chuong:
                    cur_chuong = {"chuong_so": "0", "chuong_ten": "QUY ĐỊNH CHUNG", "dieu": [], "page_number": current_page}
                    result["chuong"].append(cur_chuong)

                cur_dieu = {
                    "dieu_so": m_dieu.group(1),
                    "noi_dung": m_dieu.group(2).strip(),
                    "khoan": [],
                    "page_number": current_page
                }
                cur_chuong["dieu"].append(cur_dieu)
                cur_khoan = None; cur_diem = None; last_node = cur_dieu
                continue

            # --- KHOẢN ---
            if cur_dieu:
                m_khoan = self.re_khoan.match(line)
                if m_khoan:
                    cur_khoan = {
                        "khoan_so": m_khoan.group(1),
                        "noi_dung": m_khoan.group(2).strip(),
                        "diem": [],
                        "page_number": current_page
                    }
                    cur_dieu["khoan"].append(cur_khoan)
                    cur_diem = None; last_node = cur_khoan
                    
                    m_diem_inline = self.re_diem_split.search(cur_khoan["noi_dung"])
                    if m_diem_inline:
                        prefix = cur_khoan["noi_dung"][:m_diem_inline.start(1)-1]
                        prefix = re.sub(r'[:;.]\s*$', '', prefix).strip()
                        cur_khoan["noi_dung"] = prefix 
                        cur_diem = {
                            "diem_ky_hieu": m_diem_inline.group(1),
                            "noi_dung": m_diem_inline.group(2).strip(),
                            "page_number": current_page
                        }
                        cur_khoan["diem"].append(cur_diem)
                        last_node = cur_diem
                    continue

            # --- ĐIỂM ---
            if cur_dieu:
                m_diem = self.re_diem.match(line)
                if m_diem:
                    cur_diem = {
                        "diem_ky_hieu": m_diem.group(1),
                        "noi_dung": m_diem.group(2).strip(),
                        "page_number": current_page
                    }
                    if cur_khoan: cur_khoan["diem"].append(cur_diem)
                    last_node = cur_diem
                    continue

            # --- NỐI DÒNG ---
            if last_node:
                # Chỉ nối nếu KHÔNG phải là Header đang build (trừ khi header đã xong)
                if not finished_doc_header and not cur_chuong and not cur_dieu:
                    pass # Đang build header, không nối vào đâu cả
                else:
                    m_split = self.re_diem_split.search(line)
                    if m_split and cur_khoan: 
                        prefix_text = line[:m_split.start(1)]
                        diem_char = m_split.group(1)
                        diem_content = m_split.group(2)
                        
                        if prefix_text.strip(): self._append_text(last_node, prefix_text)
                        
                        cur_diem = {
                            "diem_ky_hieu": diem_char,
                            "noi_dung": diem_content.strip(),
                            "page_number": current_page
                        }
                        cur_khoan["diem"].append(cur_diem)
                        last_node = cur_diem
                    else:
                        self._append_text(last_node, line)
        
        if not result["ten_van_ban"]: result["ten_van_ban"] = "Không xác định"
        return result