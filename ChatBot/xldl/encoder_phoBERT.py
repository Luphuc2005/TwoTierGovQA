import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# =========================
# CONFIG
# =========================
MODEL_NAME = "vinai/phobert-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_JSON = r"D:\GitHub\ChatBot\output_nghidinh\chunks_clean.json"          # <-- file input của bạn
OUT_DIR = os.path.join("vector_data", "phobert")   # thư mục output riêng

MAX_SEQ_LENGTH = 256
OVERLAP = 32

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print(f"Loading PhoBERT on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# =========================
# EMBEDDING FUNCTIONS
# =========================
def split_and_embed(text: str, max_seq_length=256, overlap=32):
    """
    Sliding-window embedding with mean pooling (excluding <s>, </s>) + L2 norm.
    Returns: vectors(list[np.ndarray]), sub_texts(list[str])
    """
    if not text or not text.strip():
        return [], []

    # tokenize without special tokens
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False)
    input_ids = enc["input_ids"][0]  # [T]
    total_tokens = input_ids.size(0)

    max_tokens_content = max_seq_length - 2  # reserve <s> and </s>
    stride = max_tokens_content - overlap
    if stride <= 0:
        raise ValueError("OVERLAP must be smaller than max_tokens_content (max_seq_length-2).")

    chunks_ids = []
    if total_tokens <= max_tokens_content:
        chunks_ids.append(input_ids)
    else:
        for i in range(0, total_tokens, stride):
            chunk = input_ids[i:i + max_tokens_content]
            if chunk.numel() == 0:
                continue
            chunks_ids.append(chunk)

    vectors = []
    sub_texts = []

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    for chunk in chunks_ids:
        full_ids = torch.cat([
            torch.tensor([cls_id], dtype=torch.long),
            chunk.to(torch.long),
            torch.tensor([sep_id], dtype=torch.long)
        ], dim=0)  # [L]

        input_ids_batch = full_ids.unsqueeze(0).to(DEVICE)  # [1, L]
        attention_mask = torch.ones_like(input_ids_batch, device=DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask)

        # mean pooling excluding special tokens
        token_embeddings = outputs.last_hidden_state[:, 1:-1, :]  # [1, L-2, H]
        vec = token_embeddings.mean(dim=1).squeeze(0).cpu().numpy()  # [H]

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        vectors.append(vec)
        sub_texts.append(tokenizer.decode(chunk, skip_special_tokens=True))

    return vectors, sub_texts

# =========================
# MAIN
# =========================
print(f"Reading input: {INPUT_JSON}")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

final_embeddings = []
final_metadata = []

print(f"Processing {len(chunks)} chunks...")

for idx, chunk in enumerate(chunks):
    raw_text = chunk.get("text", "")
    if not raw_text or not raw_text.strip():
        continue

    vecs, segments = split_and_embed(raw_text, MAX_SEQ_LENGTH, OVERLAP)

    if idx % 10 == 0:
        print(f"Chunk {idx}: {len(vecs)} vectors")

    for i, vec in enumerate(vecs):
        final_embeddings.append(vec)

        meta = dict(chunk)  # copy
        meta["text"] = segments[i]
        meta["chunk_id"] = f"{idx}_{i}"
        final_metadata.append(meta)

embeddings_matrix = np.asarray(final_embeddings, dtype=np.float32)

# =========================
# SAVE OUTPUT (name from input)
# =========================
base_name = os.path.splitext(os.path.basename(INPUT_JSON))[0]  # chunks_clean
emb_path = os.path.join(OUT_DIR, f"{base_name}.embeddings.npy")
meta_path = os.path.join(OUT_DIR, f"{base_name}.metadata.json")

np.save(emb_path, embeddings_matrix)

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(final_metadata, f, ensure_ascii=False, indent=2)

print("====================================")
print(f"Saved embeddings: {emb_path}  shape={embeddings_matrix.shape}")
print(f"Saved metadata  : {meta_path}  items={len(final_metadata)}")
print("Done.")
