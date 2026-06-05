"""
Rules Base ProtonX - Module sử dụng OCR_paddle_protonX.py cho OCR
Kết hợp PaddleOCR detection + ProtonX recognition + Parser V11
"""
import os
import sys
import json
import argparse
import re
from pathlib import Path

# Cấu hình encoding UTF-8 cho Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import từ parser.py
from legal_parser import LawParser

# ==============================================================================
# 1. HÀM GHÉP TEXT CHUYÊN BIỆT (KHẮC PHỤC LỖI THỨ TỰ & RÁC)
# ==============================================================================

def merge_ocr_to_text(ocr_results):
    """
    Ghép kết quả OCR từ các trang thành một văn bản liền mạch.
    
    Các bước xử lý:
    1. Sắp xếp trang theo đúng thứ tự (1, 2, ..., 10).
    2. Loại bỏ Header/Footer rác (Số trang, tên văn bản lặp lại) TRƯỚC khi ghép.
    3. Nối các trang lại với nhau.
    """
    if not ocr_results:
        return ""

    # 1. Sắp xếp key theo số trang (tránh lỗi page_10 đứng trước page_2)
    # Key dạng "page_1", "page_2"...
    try:
        sorted_keys = sorted(ocr_results.keys(), key=lambda x: int(x.split('_')[-1]))
    except:
        sorted_keys = sorted(ocr_results.keys()) # Fallback nếu key lạ

    full_text_parts = []

    for key in sorted_keys:
        page_num = key.split('_')[-1]
        page_text = ocr_results[key].get("text", "")
        if not page_text: continue

        # 2. Làm sạch sơ bộ từng trang (Loại bỏ Header/Footer gây đứt câu)
        lines = page_text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Bỏ qua dòng chỉ có số (số trang)
            if re.match(r'^\d+$', line): continue
            # Bỏ qua dòng quá ngắn (rác)
            if len(line) < 2: continue
            # Bỏ qua Header thường gặp
            if re.match(r'^(VGP|CÔNG THÔNG TIN|Trang \d+|Page \d+)', line, re.IGNORECASE):
                continue
            
            clean_lines.append(line)
        
        # Ghép lại trang đã sạch kèm theo marker trang [PAGE_N]
        page_content = f"\n[PAGE_{page_num}]\n" + "\n".join(clean_lines)
        full_text_parts.append(page_content)

    # 3. Nối các trang
    return "\n\n".join(full_text_parts)


def count_ocr_fixes(original, fixed):
    """Đếm số lượng sửa lỗi (giữ nguyên)"""
    return 0 # Placeholder

# ==============================================================================
# 2. PIPELINE CHÍNH
# ==============================================================================

def run_full_pipeline_protonx(pdf_path, output_dir=None):
    """
    Pipeline: OCR -> Merge Text -> Parse -> Clean
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")
    
    if output_dir is None: output_dir = Path("output_protonx_pipeline")
    else: output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = pdf_path.stem
    print(f"\n{'='*70}\nPROTONX PIPELINE - {pdf_path.name}\n{'='*70}\n")
    
    # BƯỚC 1: OCR
    print("[1/4] Chạy OCR (Paddle + ProtonX)...")
    ocr_output_path = output_dir / f"{base_name}_protonx_ocr.json"
    from OCR_paddle_protonX import process_pdf as process_pdf_protonx
    ocr_results = process_pdf_protonx(str(pdf_path), str(ocr_output_path))
    
    # BƯỚC 2: GHÉP TEXT (QUY TRÌNH MỚI)
    print("\n[2/4] Ghép và làm sạch text từ các trang...")
    full_text = merge_ocr_to_text(ocr_results)
    
    # BƯỚC 3: Parse
    parser = LawParser()
    print(full_text)  # In 500 ký tự đầu để kiểm tra
    parsed_result = parser.parse(full_text)
    
    parsed_output_path = output_dir / f"{base_name}_parsed.json"
    with open(parsed_output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_result, f, ensure_ascii=False, indent=2)
    
    # BƯỚC 4: Hậu xử lý & Clean
    print("\n[4/4] Hậu xử lý và sửa lỗi...")
    final_result = parsed_result
    
    final_output_path = output_dir / f"{base_name}_final.json"
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ HOÀN THÀNH. Output: {final_output_path}")
    return final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf_path', help='Path to PDF')
    parser.add_argument('-o', '--output-dir', default='output_protonx_pipeline')
    args = parser.parse_args()
    try:
        run_full_pipeline_protonx(args.pdf_path, args.output_dir)
    except Exception as e:
        print(f"❌ Error: {e}")
