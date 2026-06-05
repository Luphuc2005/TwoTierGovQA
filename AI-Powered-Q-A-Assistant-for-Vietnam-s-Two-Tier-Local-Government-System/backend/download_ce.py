from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "BAAI/bge-reranker-v2-m3"
dest_dir = r"D:\GitHub\Folder cha\ChatBot\cross-encoder\outputs\models\ce_bge_reranker_ft_v6"

print("Downloading model.safetensors...")
try:
    file_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(f"Downloaded to {file_path}")
    
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "model.safetensors")
    shutil.copy2(file_path, dest_path)
    print(f"Copied to {dest_path}")
except Exception as e:
    print(f"Error: {e}")
