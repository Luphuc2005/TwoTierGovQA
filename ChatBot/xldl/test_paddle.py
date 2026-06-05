import os
import sys

lib_bin = r"C:\Users\ADMIN\anaconda3\envs\ChatBot\Library\bin"
if os.path.exists(lib_bin):
    os.add_dll_directory(lib_bin)
import torch

from pdf2image import convert_from_path
import numpy as np
import cv2
from paddleocr import PaddleOCR

POPPLER_PATH = r"D:\Release-25.12.0-0\poppler-25.12.0\Library\bin"
pdf_path = r"D:\GitHub\Folder cha - Copy\AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System\backend\uploads\decrees\202qh.signed_sapxepdonvihanhchinh.pdf"

print("Extracting first page...")
images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1, poppler_path=POPPLER_PATH, fmt='jpeg')
image = images[0]

img_cv = np.array(image)
if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

print("Initializing PaddleOCR...")
paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='vi',
    use_gpu=False,
    show_log=False,
    det_db_thresh=0.1,
    det_db_box_thresh=0.3,
    det_db_unclip_ratio=2.0,
    det_limit_side_len=2500,
    det_limit_type='max',
    use_dilation=True,
    det_db_score_mode='fast'
)

print("Running OCR...")
result = paddle_ocr.ocr(img_cv, cls=True)

print("Result:", result)
print("Finished!")
