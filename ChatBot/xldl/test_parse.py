import json
import os
import sys

# D:\GitHub\Folder cha - Copy\ChatBot\xldl
sys.path.append(r"D:\GitHub\Folder cha - Copy\ChatBot\xldl")

from legal_parser import LawParser
from rules_base_protonx import merge_ocr_to_text

ocr_json = r"D:\GitHub\Folder cha - Copy\AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System\backend\uploads\decrees\processed\202qh.signed_sapxepdonvihanhchinh_protonx_ocr.json"

with open(ocr_json, 'r', encoding='utf-8') as f:
    ocr_results_dict = json.load(f)

full_text = merge_ocr_to_text(ocr_results_dict)
print("--- Đầu văn bản ---")
print(full_text[:500])
print("...")
print("--- Parse ---")
parser = LawParser()
parsed_result = parser.parse(full_text)

print("Tên văn bản:", parsed_result.get("ten_van_ban"))
print("Số chương:", len(parsed_result.get("chuong", [])))
if parsed_result.get("chuong"):
    print("Các điều trong chương 1:", len(parsed_result["chuong"][0].get("dieu", [])))

with open("test_parse_output.json", "w", encoding='utf-8') as f:
    json.dump(parsed_result, f, ensure_ascii=False, indent=2)
print("Done!")
