import os
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm.auto import tqdm

# Token và cấu hình
HF_TOKEN = "" # [CHÚ Ý] Điền Hugging Face Token của bạn vào đây nếu cần tải model.
models = {
    "Irthn1311/bi-bge-m3-legal-vn": "outputs/models/bi_bge_m3_ft",
    "Irthn1311/ce-bge-reranker-legal-vn-v6": "outputs/models/ce_bge_reranker_ft_v6"
}

base_dir = os.path.dirname(os.path.abspath(__file__))

for repo_id, relative_path in models.items():
    local_dir = os.path.join(base_dir, relative_path)
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"\n[TIẾN TRÌNH] Đang tải mô hình: {repo_id}")
    files = list_repo_files(repo_id, token=HF_TOKEN)
    
    for file in files:
        print(f"  -> Đang tải: {file}")
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            local_dir=local_dir,
            token=HF_TOKEN,
            local_dir_use_symlinks=False
        )

print("\n--- ✅ TẤT CẢ ĐÃ TẢI XONG! ---")
