"""
COMPLETE PIPELINE WITH PROTONX OCR
===================================
PDF → OCR (PaddleOCR + ProtonX) → Parsing → OCR Fixes → Chunking → LLM (optional)
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
# Cấu hình encoding UTF-8 cho Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from load_env import load_env, get_env
    load_env()
except ImportError:
    def get_env(key: str, default: str = None) -> str:
        return os.getenv(key, default)


class CompletePipelineProtonX:
    """Complete pipeline using ProtonX OCR"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.use_llm = self.config.get('use_llm', False)
        self.llm_model = self.config.get('llm_model', get_env('LLM_MODEL', 'gemini-1.5-flash-latest'))
        self.output_dir = self.config.get('output_dir', 'output_protonx_complete')
        
    def step1_ocr_and_parse(self, pdf_path: str, output_dir: str) -> Dict:
        """STEP 1: OCR & Parsing với ProtonX"""
        print(f"\n{'='*70}")
        print(f"STEP 1: OCR & Parsing (ProtonX)")
        print(f"{'='*70}")
        
        from rules_base_protonx import run_full_pipeline_protonx
        
        parsed_data = run_full_pipeline_protonx(pdf_path, output_dir=output_dir)
        parsed_data['source_file'] = pdf_path
        
        return parsed_data
    
    def step2_chunk(self, parsed_data: Dict, output_dir: str) -> Dict:
        """STEP 2: Chunking"""
        print(f"\n{'='*70}")
        print(f"STEP 2: CHUNKING")
        print(f"{'='*70}")
        
        chunks = self._chunk_parsed_data(parsed_data)
        
        chunks_path = os.path.join(output_dir, f"{Path(parsed_data['source_file']).stem}_chunks.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump({'chunks': chunks}, f, ensure_ascii=False, indent=2)
        
        print(f"   ✓ Created: {len(chunks)} chunks")
        print(f"   ✓ Saved: {chunks_path}")
        
        return {'chunks': chunks, 'chunks_path': chunks_path}
    
    def _chunk_parsed_data(self, parsed_data: Dict) -> List[Dict]:
        """Chunk parsed data into Khoản (Clause) or Điểm (Point) level for better RAG granularity"""
        chunks = []
        chunk_id = 1
        source_file = parsed_data.get('source_file', '')
        
        # 1. Metadata chunk
        metadata = parsed_data.get('metadata', {})
        ten_vb = parsed_data.get('ten_van_ban') or metadata.get('loai_van_ban', '') + " " + metadata.get('so_hieu', '')
        ten_vb = ten_vb.strip()
        
        meta_text = f"{ten_vb}\n"
        meta_text += f"Cơ quan ban hành: {metadata.get('co_quan_ban_hanh', '')}\n"
        meta_text += f"Ngày ban hành: {metadata.get('ngay_ban_hanh', '')}"
        
        chunks.append({
            'id': f'chunk_{chunk_id}',
            'text': meta_text.strip(),
            'metadata': {
                'type': 'metadata', 
                'van_ban': ten_vb, 
                'source_file': source_file,
                **metadata
            }
        })
        chunk_id += 1
        
        # 2. Căn cứ pháp lý
        for i, cc in enumerate(parsed_data.get('can_cu_phap_ly', [])):
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': f"Căn cứ {i+1} của {ten_vb}: {cc}",
                'metadata': {
                    'type': 'can_cu', 
                    'van_ban': ten_vb,
                    'source_file': source_file
                }
            })
            chunk_id += 1
        
        # 3. Chapters and Articles
        for chuong in parsed_data.get('chuong', []):
            c_so = chuong.get('chuong_so', chuong.get('so_chuong', ''))
            c_ten = chuong.get('chuong_ten', chuong.get('ten_chuong', ''))
            
            for dieu in chuong.get('dieu', []):
                d_so = dieu.get('dieu_so', dieu.get('so_dieu', ''))
                d_noi_dung = dieu.get('noi_dung', '')
                
                # If no clauses, chunk at the Article level
                if not dieu.get('khoan'):
                    text = f"Theo Điều {d_so} của {ten_vb}: {d_noi_dung}"
                    if c_so: text = f"Chương {c_so} ({c_ten}). {text}"
                    
                    chunks.append({
                        'id': f'chunk_{chunk_id}',
                        'text': text.strip(),
                        'metadata': {
                            'type': 'dieu',
                            'van_ban': ten_vb,
                            'chuong': c_so,
                            'dieu': d_so,
                            'page_number': dieu.get('page_number'),
                            'source_file': source_file
                        }
                    })
                    chunk_id += 1
                    continue

                # Iterate through Clauses (Khoản)
                for khoan in dieu.get('khoan', []):
                    k_so = khoan.get('khoan_so', khoan.get('so_khoan', ''))
                    k_noi_dung = khoan.get('noi_dung', '')
                    
                    # If has Points (Điểm), chunk at the Point level
                    if khoan.get('diem'):
                        for diem in khoan['diem']:
                            diem_ky_hieu = diem.get('diem_ky_hieu', diem.get('so_diem', ''))
                            diem_nd = diem.get('noi_dung', '')
                            
                            text = f"Tại điểm {diem_ky_hieu}, Khoản {k_so}, Điều {d_so} của {ten_vb}: {diem_nd}"
                            if c_so: text = f"Chương {c_so}. {text}"
                            
                            chunks.append({
                                'id': f'chunk_{chunk_id}',
                                'text': text.strip(),
                                'metadata': {
                                    'type': 'diem',
                                    'van_ban': ten_vb,
                                    'chuong': c_so,
                                    'dieu': d_so,
                                    'khoan': k_so,
                                    'diem': diem_ky_hieu,
                                    'page_number': diem.get('page_number'),
                                    'source_file': source_file
                                }
                            })
                            chunk_id += 1
                    else:
                        # Chunk at Clause (Khoản) level
                        text = f"Khoản {k_so}, Điều {d_so} của {ten_vb}: {k_noi_dung}"
                        if c_so: text = f"Chương {c_so}. {text}"
                        
                        chunks.append({
                            'id': f'chunk_{chunk_id}',
                            'text': text.strip(),
                            'metadata': {
                                'type': 'khoan',
                                'van_ban': ten_vb,
                                'chuong': c_so,
                                'dieu': d_so,
                                'khoan': k_so,
                                'page_number': khoan.get('page_number'),
                                'source_file': source_file
                            }
                        })
                        chunk_id += 1
        
        return chunks
    
    def run(self, pdf_path: str, only_parse: bool = False, only_index: bool = False, append: bool = False) -> Dict:
        """Chạy pipeline với các tùy chọn bước"""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"COMPLETE PROTONX PIPELINE")
        print(f"Input: {pdf_path}")
        print(f"Output: {output_dir}")
        print(f"Only Parse: {only_parse} | Only Index: {only_index} | Append: {append}")
        print(f"{'='*70}")

        parsed_data = None
        chunk_result = None

        if not only_index:
            # Step 1: OCR & Parse
            parsed_data = self.step1_ocr_and_parse(pdf_path, str(output_dir))
            
            # Step 2: Chunk
            chunk_result = self.step2_chunk(parsed_data, str(output_dir))
            
            if only_parse:
                print("\n✅ Step 1 & 2 completed. Skipping Step 3 as requested.")
                return {'parsed': parsed_data, 'chunks': chunk_result['chunks'], 'output_dir': str(output_dir)}
        else:
            # Load existing chunks if only indexing
            chunks_path = output_dir / f"{Path(pdf_path).stem}_chunks.json"
            if chunks_path.exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunk_result = json.load(f)
                print(f"✅ Loaded existing chunks from {chunks_path}")
                # Mock parsed_data for metadata consistency
                parsed_data = {'source_file': pdf_path}
            else:
                raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        # Step 3: FAISS
        print(f"\n{'='*70}")
        print(f"STEP 3: FAISS VECTOR DB INGESTION")
        print(f"{'='*70}")
        faiss_success = False
        faiss_total = 0
        
        # Determine paths relative to root
        root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        prod_dir = root_dir / "vector_data" / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)
        
        idx_path = prod_dir / "index.faiss"
        meta_path = prod_dir / "metadata.json"
        
        print(f"   📄 Chunks source: {len(chunk_result['chunks'])} chunks")
        print(f"   🎯 Target: {prod_dir}")
        
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            # 1. Load Model (Same as rag_service.py)
            model_path = str(root_dir / "ChatBot" / "cross-encoder" / "outputs" / "models" / "bi_bge_m3_ft")
            print(f"   📦 Loading model: {model_path}...")
            if not os.path.isdir(model_path):
                # Fallback to public model if local ft not found
                print(f"   ⚠️ Local model not found, falling back to BAAI/bge-m3")
                model_path = "BAAI/bge-m3"
            
            model = SentenceTransformer(model_path)
            
            # 2. Extract texts
            texts = [c['text'] for c in chunk_result['chunks']]
            
            # 3. Encode
            print(f"   🔄 Encoding {len(texts)} chunks (this might take a while)...")
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            
            # 4. Build or Load Index
            dim = embeddings.shape[1]
            existing_chunks = []
            
            if append and idx_path.exists() and meta_path.exists():
                print(f"   📂 Appending to existing index at {idx_path}")
                index = faiss.read_index(str(idx_path))
                with open(meta_path, 'r', encoding='utf-8') as f:
                    existing_chunks = json.load(f)
                print(f"   📦 Existing chunks: {len(existing_chunks)}")
            else:
                print(f"   🆕 Creating NEW index")
                index = faiss.IndexFlatIP(dim)
            
            # Add new embeddings
            index.add(embeddings)
            
            # 5. Save
            print(f"   💾 Saving combined index ({index.ntotal} vectors)...")
            faiss.write_index(index, str(idx_path))
            
            # Combine and save metadata
            all_chunks = existing_chunks + chunk_result['chunks']
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                
            faiss_success = True
            faiss_total = index.ntotal
            print(f"   ✓ FAISS ingestion successful!")
            print(f"   ✓ Index saved to: {idx_path}")
            print(f"   ✓ Metadata saved to: {meta_path}")
            
        except Exception as e:
            print(f"   ✗ FAISS ingestion failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        print(f"\n{'='*70}")
        print("✅ PIPELINE COMPLETED")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")
        print(f"Total chunks: {len(chunk_result['chunks'])}")
        print(f"FAISS Indexed: {'✓ Yes' if faiss_success else '✗ No (failed)'}")
        if faiss_total > 0:
            print(f"FAISS Total Vectors: {faiss_total}")
        print(f"{'='*70}\n")
        
        return {
            'parsed': parsed_data,
            'chunks': chunk_result['chunks'],
            'output_dir': str(output_dir),
            'vector_dir': str(prod_dir)
        }


def main():
    parser = argparse.ArgumentParser(description='Complete Pipeline với ProtonX OCR')
    parser.add_argument('pdf_path', help='Đường dẫn đến file PDF')
    parser.add_argument('-o', '--output-dir', default='output_protonx_complete',
                        help='Thư mục lưu kết quả')
    parser.add_argument('--use-llm', action='store_true',
                        help='Sử dụng LLM để chuẩn hóa (chưa implement)')
    parser.add_argument('--only-parse', action='store_true',
                        help='Chỉ chạy OCR và Parsing, không nạp FAISS')
    parser.add_argument('--only-index', action='store_true',
                        help='Chỉ chạy nạp FAISS từ file chunks đã có')
    parser.add_argument('--append', action='store_true',
                        help='Nạp thêm vào index hiện có thay vì ghi đè')
    
    args = parser.parse_args()
    
    config = {
        'output_dir': args.output_dir,
        'use_llm': args.use_llm
    }
    
    pipeline = CompletePipelineProtonX(config)
    
    try:
        result = pipeline.run(
            args.pdf_path, 
            only_parse=args.only_parse, 
            only_index=args.only_index, 
            append=args.append
        )
        print(f"\n✅ Hoàn thành! Kết quả tại: {result['output_dir']}")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
