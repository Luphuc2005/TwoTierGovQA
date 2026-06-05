import torch
from safetensors.torch import save_file
import os

weight_path = r"D:\GitHub\Folder cha\ChatBot\cross-encoder\outputs\models\bi_bge_m3_ft\pytorch_model.bin"
out_path = r"D:\GitHub\Folder cha\ChatBot\cross-encoder\outputs\models\bi_bge_m3_ft\model.safetensors"

if os.path.exists(weight_path):
    try:
        print("Loading bin...")
        state_dict = torch.load(weight_path, map_location="cpu")
        print("Saving safetensors...")
        save_file(state_dict, out_path)
        os.remove(weight_path)
        print("Converted bi-encoder weights to safetensors.")
    except Exception as e:
        print("Error:", e)
else:
    print("Bin file not found.")
