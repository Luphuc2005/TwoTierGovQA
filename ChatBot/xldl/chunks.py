import glob
import json
import os

def build_chunks_from_file(path):
    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        vb = json.load(f)

    ten_vb = vb.get("ten_van_ban", "")
    print(f"🔹 Xử lý văn bản: {ten_vb}")

    for chuong in vb.get("chuong", []):
        chuong_so = chuong.get("chuong_so")

        for dieu in chuong.get("dieu", []):
            dieu_so = dieu.get("dieu_so")
            dieu_nd = dieu.get("noi_dung", "")
            if not dieu.get("khoan"):
                # 🔹 Không có khoản → chunk ở mức điều
                text = (
                    f"Theo Điều {dieu_so} {ten_vb} {dieu_nd}"
                )

                chunks.append({
                    "text": text,
                    "metadata": {
                        "van_ban": ten_vb,
                        "chuong": chuong_so,
                        "dieu": dieu_so,
                        "khoan": None,
                        "diem": None,
                        "source_file": path
                    }
                })
                continue    

            for khoan in dieu.get("khoan", []):
                khoan_so = khoan.get("khoan_so")
                khoan_nd = khoan.get("noi_dung", "")

                # 🔹 Nếu có điểm → chunk theo điểm
                if khoan.get("diem"):
                    for diem in khoan["diem"]:
                        diem_ky_hieu = diem.get("diem_ky_hieu")
                        diem_nd = diem.get("noi_dung", "")

                        context_prefix = f" ({khoan_nd})" if khoan_nd else ""
                        text = (
                            f"Tại điểm {diem_ky_hieu} Khoản {khoan_so} Điều {dieu_so} {ten_vb}{context_prefix} {diem_nd}"
                        )

                        chunks.append({
                            "text": text,
                            "metadata": {
                                "van_ban": ten_vb,
                                "chuong": chuong_so,
                                "dieu": dieu_so,
                                "khoan": khoan_so,
                                "diem": diem_ky_hieu,
                                "source_file": path
                            }
                        })

                # 🔹 Không có điểm → chunk ở mức khoản
                else:
                    text = (
                        f"Khoản {khoan_so} Điều {dieu_so} {ten_vb} {khoan_nd}"
                    )

                    chunks.append({
                        "text": text,
                        "metadata": {
                            "van_ban": ten_vb,
                            "chuong": chuong_so,
                            "dieu": dieu_so,
                            "khoan": khoan_so,
                            "diem": None,
                            "source_file": path
                        }
                    })

    return chunks

if __name__ == "__main__":
    from load_env import load_env, get_env
    load_env()
    folder_path = "./output_nghidinh"
    all_data = []
    chunky = []

    for file_path in glob.glob(os.path.join(folder_path, "*final.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                all_data.append(data)
            except json.JSONDecodeError:
                print("❌ Lỗi JSON:", file_path)

    for file_path in glob.glob(os.path.join(folder_path, "*final.json")):
        file_chunks = build_chunks_from_file(file_path)
        chunky.extend(file_chunks)
    output_path = get_env("FILE_CHUNKS")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunky, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu {len(chunky)} chunks vào {output_path}")
    print(f"✅ Đã xử lý {len(all_data)} file JSON.")
    print(f"✅ Tổng số chunk tạo được: {len(chunky)}")

