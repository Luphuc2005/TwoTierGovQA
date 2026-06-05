import sys
from pathlib import Path
import json

# Setup paths to match rag_service.py logic
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
_FOLDER_CHA = _PROJECT_DIR.parent
_CHATBOT_ROOT = _FOLDER_CHA / "ChatBot"

if str(_CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHATBOT_ROOT))

from generation.run_generation import LegalRetriever
from generation.retrieval_rerank import RetrievalReranker

def test_retrieval():
    query = "Bộ Tư pháp là cơ quan của Chính phủ thực hiện chức năng quản lý nhà nước về những lĩnh vực nào?"
    print(f"Retrieving for: {query}\n")

    print("--- Testing LegalRetriever ---")
    retriever_legal = LegalRetriever(device="cpu")
    chunks_legal = retriever_legal.retrieve_and_rerank(query, top_k_rerank=3)
    for i, c in enumerate(chunks_legal):
        print(f"[Legal {i+1}] Score: {c.score_rerank:.4f} | Source: {c.van_ban}")

    print("\n--- Testing RetrievalReranker ---")
    retriever_rr = RetrievalReranker(device="cpu")
    chunks_rr = retriever_rr.run(query, top_k=3)
    for i, c in enumerate(chunks_rr):
        print(f"[RR {i+1}] Score: {c['ce_score']:.4f} | Source: {c['van_ban']}")

if __name__ == "__main__":
    test_retrieval()
