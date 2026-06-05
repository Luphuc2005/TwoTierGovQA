import json
import re
from pathlib import Path

IN_PATH  = Path("output_nghidinh/chunks.json")
OUT_PATH = Path("output_nghidinh/chunks_clean.json")

def normalize_after_remove(s: str) -> str:
    # gộp khoảng trắng
    s = re.sub(r"\s+", " ", s).strip()

    # dọn khoảng trắng trước dấu câu
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)

    # nếu sau khi xóa bị dính dấu câu kiểu ", ," hoặc "; ;"
    s = re.sub(r"([,.;:!?])\s*\1+", r"\1", s)

    # nếu bị kiểu "Tư pháp ;" -> "Tư pháp;"
    s = re.sub(r"\s+([;:])", r"\1", s)

    # dọn khoảng trắng sau mở ngoặc / trước đóng ngoặc
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)

    return s

def remove_van_ban_from_text(text: str, van_ban: str) -> tuple[str, bool]:
    if not text or not van_ban:
        return text, False

    # Xóa đúng chuỗi van_ban (match literal)
    if van_ban in text:
        new_text = text.replace(van_ban, "")
        new_text = normalize_after_remove(new_text)
        return new_text, True

    return text, False

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Không thấy file: {IN_PATH.resolve()}")

    chunks = json.loads(IN_PATH.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError("chunks.json phải là một LIST các object chunk")

    changed = 0
    examples = []

    for c in chunks:
        text = c.get("text", "")
        meta = c.get("metadata", {}) or {}
        vb = meta.get("van_ban", "")

        new_text, did = remove_van_ban_from_text(text, vb)
        if did:
            changed += 1
            # lưu vài ví dụ để in ra
            if len(examples) < 5:
                examples.append((vb, text, new_text))

        c["text"] = new_text

    OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Done")
    print(f"Input : {IN_PATH} | chunks = {len(chunks)}")
    print(f"Output: {OUT_PATH}")
    print(f"Changed chunks: {changed}")

    if examples:
        print("\n--- SAMPLE BEFORE/AFTER (tối đa 5) ---")
        for i, (vb, before, after) in enumerate(examples, 1):
            print(f"\n#{i}")
            print(f"van_ban: {vb}")
            print("BEFORE:", before)
            print("AFTER :", after)

if __name__ == "__main__":
    main()