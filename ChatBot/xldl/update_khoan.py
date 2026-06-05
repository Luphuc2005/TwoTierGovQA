import json

INP = "output_nghidinh/chunks_clean.json"
OUT = "output_nghidinh/chunks_clean_norm.json"

data = json.load(open(INP, "r", encoding="utf-8"))
if isinstance(data, dict) and "chunks" in data:
    chunks = data["chunks"]
else:
    chunks = data

for c in chunks:
    meta = c.get("metadata", {})
    if meta.get("khoan", "") is None:
        meta["khoan"] = ""
    # optional: chuẩn hóa dieu/khoan về string
    if "dieu" in meta and meta["dieu"] is not None:
        meta["dieu"] = str(meta["dieu"]).strip()
    if "khoan" in meta and meta["khoan"] is not None:
        meta["khoan"] = str(meta["khoan"]).strip()

json.dump(chunks, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Saved:", OUT, "N=", len(chunks))