"""
OCR Pipeline: PaddleOCR (Detection) + VietOCR (Recognition) + ProtonX (Text Correction)
- PaddleOCR: Phát hiện vùng text và nhận dạng sơ bộ.
- VietOCR: Nhận dạng lại text tiếng Việt từ các vùng đã detect (chính xác hơn).
- ProtonX: Sửa lỗi OCR.
Optimized for low memory usage & Robust Error Handling.
"""
import os
import sys

# WORKAROUND FOR WINERROR 127: Import torch FIRST!
# cv2 and paddleocr bundle their own C++ DLLs (like libiomp5md.dll). If they are imported first,
# they load their DLLs into memory. When torch is imported later, it fails because of DLL conflicts.
# Also explicitly add the Conda Library\bin to the Windows DLL search path:
if sys.platform == 'win32':
    try:
        # Using the exact Conda env path
        lib_bin = r"C:\Users\ADMIN\anaconda3\envs\ChatBot\Library\bin"
        if os.path.exists(lib_bin):
            os.add_dll_directory(lib_bin)
    except Exception:
        pass

import torch

import json
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from tqdm import tqdm
import gc
import pdfplumber

# Cấu hình encoding UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Đổi cache HuggingFace sang ổ D để tránh đầy ổ C
# Ưu tiên đọc từ biến môi trường, fallback về giá trị mặc định
_hf_home = os.environ.get('HF_HOME', 'D:/huggingface_cache')
os.environ['HF_HOME'] = _hf_home
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', _hf_home)

# === CẤU HÌNH ===
POPPLER_PATH = os.environ.get('POPPLER_PATH', r"D:\Release-25.12.0-0\poppler-25.12.0\Library\bin")
MODEL_NAME = "protonx-models/protonx-legal-tc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("OCR Pipeline: PaddleOCR + VietOCR + ProtonX")
print("=" * 60)

# ------------------------------------------------------------------------------
# 1. LOAD PADDLE OCR
# ------------------------------------------------------------------------------
print("\n1. Loading PaddleOCR for text detection...")
try:
    paddle_ocr = PaddleOCR(
        use_angle_cls=True,
        lang='vi',
        use_gpu=True,
        show_log=False,
        det_db_thresh=0.1,       # Ngưỡng thấp để bắt text mờ
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=2.0,
        det_limit_side_len=2500, # Tăng size xử lý
        det_limit_type='max',
        use_dilation=True,
        det_db_score_mode='fast'
    )
    print("   ✓ PaddleOCR loaded")
except Exception as e:
    print(f"   ✗ Error loading PaddleOCR: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. LOAD VIETOCR (Đã thêm lại phần này)
# ------------------------------------------------------------------------------
print("\n2. Loading VietOCR for text recognition...")
VIETOCR_AVAILABLE = False
vietocr_predictor = None

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    
    # Load VietOCR config (vgg_transformer hoặc vgg_seq2seq)
    config = Cfg.load_config_from_name('vgg_transformer') 
    config['cnn']['pretrained'] = True
    config['device'] = DEVICE
    config['predictor']['beamsearch'] = False # Tắt beamsearch cho nhanh
    
    vietocr_predictor = Predictor(config)
    VIETOCR_AVAILABLE = True
    print(f"   ✓ VietOCR loaded on {DEVICE}")
except ImportError:
    print("   ✗ VietOCR not installed. Install with: pip install vietocr")
    print("   -> Sẽ chỉ dùng PaddleOCR (kém chính xác hơn với tiếng Việt).")
except Exception as e:
    print(f"   ✗ Error loading VietOCR: {e}")
    print("   -> Sẽ chỉ dùng PaddleOCR.")

# ------------------------------------------------------------------------------
# 3. LOAD PROTONX (DISABLED)
# ------------------------------------------------------------------------------
print("\n3. Loading ProtonX Text Correction model...")
PROTONX_AVAILABLE = False
protonx_tokenizer = None
protonx_model = None

# Bypassing ProtonX because its corrupted vocabulary mismatches with mt5-base,
# causing irrecoverable CUDA device-side asserts (torch.cuda.OutOfMemory/IndexError)
print("   -> Bypassing ProtonX because of broken tokenizer and CUDA asserts. Using Paddle+VietOCR directly.")


# ==============================================================================
# CÁC HÀM XỬ LÝ CHÍNH
# ==============================================================================

def crop_box_from_image(image, box):
    """Cắt vùng ảnh chứa chữ dựa trên box (Hỗ trợ cho VietOCR)"""
    try:
        box = np.array(box, dtype=np.float32)
        
        # Padding nhẹ
        padding = 2
        x_min = int(max(0, min(box[:, 0]) - padding))
        x_max = int(min(image.shape[1], max(box[:, 0]) + padding))
        y_min = int(max(0, min(box[:, 1]) - padding))
        y_max = int(min(image.shape[0], max(box[:, 1]) + padding))
        
        if x_max <= x_min or y_max <= y_min:
            return None
            
        cropped = image[y_min:y_max, x_min:x_max]
        if cropped.size == 0: return None
        
        # Convert sang PIL Image (RGB) cho VietOCR
        if len(cropped.shape) == 3 and cropped.shape[2] == 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped)
    except Exception:
        return None

def correct_text_with_protonx(text, max_tokens=512):
    """Sử dụng ProtonX model để sửa lỗi OCR trong text"""
    # Type check an toàn
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    
    if not PROTONX_AVAILABLE or len(text.strip()) < 3:
        return text
    
    # Fallback: Nếu text quá dài, trả về gốc để tránh bị cắt
    if len(text.split()) > 400:
        return text

    try:
        inputs = protonx_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = protonx_model.generate(
                **inputs,
                num_beams=3,
                num_return_sequences=1,
                max_new_tokens=max_tokens,
                early_stopping=True,
                repetition_penalty=1.2,    # Chống lặp từ
                no_repeat_ngram_size=3     # Chống lặp cụm từ
            )
        
        corrected = protonx_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Fallback an toàn nếu kết quả quá khác biệt
        if len(corrected) < len(text) * 0.5 or len(corrected) > len(text) * 1.5:
            return text
            
        return corrected.strip()
    except Exception as e:
        return text

def detect_and_recognize_text(image):
    """
    Sử dụng PaddleOCR để phát hiện và nhận dạng text, xử lý các trường hợp output dị.
    Sửa lỗi format nested-list của PaddleOCR 2.8+
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    result = paddle_ocr.ocr(image, cls=True)
    if not result:
        return []
        
    results = []
    
    # Hàm đệ quy để moi từng [box, (text, conf)] ra khỏi n tầng array
    def extract_boxes(node):
        if not isinstance(node, list):
            return
        if len(node) == 2 and isinstance(node[0], list) and isinstance(node[1], tuple):
            # Format chuẩn: [box_points_list, (text, conf)]
            if len(node[0]) == 4:
                box = node[0]
                text_tuple = node[1]
                text = str(text_tuple[0])
                conf = float(text_tuple[1]) if len(text_tuple) > 1 else 1.0
                results.append({
                    "box": box if isinstance(box, list) else box.tolist(),
                    "text": text,
                    "confidence": conf
                })
            return
        # Nếu chưa tới độ sâu cuối, duyệt tiếp
        for item in node:
            if isinstance(item, list):
                extract_boxes(item)
                
    extract_boxes(result)
    return results

def ocr_image_hybrid(image_path):
    """
    Pipeline OCR Hợp nhất: 
    1. Detect bằng Paddle (dùng logic source của bạn) 
    2. Recognize lại bằng VietOCR (nếu có)
    3. Correct bằng ProtonX
    """
    print(f"\nProcessing: {image_path}")
    
    if isinstance(image_path, str):
        img_cv = cv2.imread(image_path)
    else:
        img_pil = image_path
        img_cv = np.array(image_path)
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Resize ảnh lớn
    max_dim = 2800
    h, w = img_cv.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"   Resizing from {w}x{h} to {new_w}x{new_h} (saving RAM)...")
        img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    gc.collect()
    
    # BƯỚC 1: DETECT & INITIAL RECOGNIZE (PADDLE)
    print("   1. Detecting and recognizing text with PaddleOCR...")
    ocr_results = detect_and_recognize_text(img_cv)
    print(f"      Found {len(ocr_results)} text boxes")
    
    # BƯỚC 2: RECOGNIZE REFINEMENT (VIETOCR)
    # Đây là phần bạn muốn thêm vào
    if VIETOCR_AVAILABLE and len(ocr_results) > 0:
        print("   2. Improving accuracy with VietOCR...")
        count_improved = 0
        for item in tqdm(ocr_results, desc="      VietOCR"):
            box = item["box"]
            
            # Cắt ảnh từ box
            cropped_img = crop_box_from_image(img_cv, box)
            
            if cropped_img is not None:
                try:
                    # Đọc lại bằng VietOCR
                    viet_text = vietocr_predictor.predict(cropped_img)
                    
                    # Nếu VietOCR đọc ra chữ có nghĩa, dùng nó thay thế Paddle
                    if viet_text and len(viet_text.strip()) > 0:
                        # Logic nhỏ: Nếu Paddle đọc ra số năm (2025) mà VietOCR đọc sai thì giữ Paddle
                        # Nhưng nhìn chung VietOCR tốt hơn với dấu tiếng Việt
                        item["paddle_text"] = item["text"] # Lưu lại để tham khảo
                        item["text"] = viet_text
                        count_improved += 1
                except Exception:
                    pass
        print(f"      -> VietOCR processed {count_improved} boxes.")

    # BƯỚC 3: CORRECTION (PROTONX)
    if PROTONX_AVAILABLE and len(ocr_results) > 0:
        print("   3. Correcting OCR errors with ProtonX...")
        for item in tqdm(ocr_results, desc="      Correcting"):
            if item["text"] and len(item["text"].split()) > 3: # Chỉ sửa câu dài
                item["text_original"] = item["text"]
                item["text"] = correct_text_with_protonx(item["text"])
    else:
        print("   3. ProtonX not available or no text found, skipping correction")
    
    return ocr_results

def process_pdf(pdf_path, output_path=None):
    """Xử lý toàn bộ PDF - tối ưu memory"""
    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_path}")
    print(f"{'='*60}")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    
    print(f"\nTotal pages: {total_pages}")
    print("Processing page by page to save memory...\n")
    
    all_results = {}
    for page_num in range(1, total_pages + 1):
        print(f"\n--- Page {page_num}/{total_pages} ---")
        
        kwargs = {'dpi': 300, 'first_page': page_num, 'last_page': page_num, 'fmt': 'jpeg'}
        if os.path.exists(POPPLER_PATH):
            kwargs['poppler_path'] = POPPLER_PATH
            
        try:
            images = convert_from_path(pdf_path, **kwargs)
        except Exception as e:
            print(f"Error converting page {page_num}: {e}")
            continue
            
        if not images: continue
            
        image = images[0]
        results = ocr_image_hybrid(image)
        
        # Sắp xếp lại box cho đúng thứ tự đọc (Trên xuống dưới, Trái sang phải)
        results.sort(key=lambda x: (x['box'][0][1], x['box'][0][0]))
        
        page_text = "\n".join([r["text"] for r in results if r.get("text")])
        all_results[f"page_{page_num}"] = {
            "page_number": page_num,
            "text": page_text,
            "boxes": results
        }
        
        del images
        del image
        del results
        gc.collect()
        
        if page_num % 10 == 0:
            print(f"\n>>> Clearing memory cache (page {page_num})...")
            gc.collect()
            import time
            time.sleep(0.5)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    # Test script
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        out_file = pdf_file.replace('.pdf', '_protonx_ocr.json')
        process_pdf(pdf_file, out_file)
    else:
        print("Usage: python OCR_paddle_protonX.py <pdf_path>")