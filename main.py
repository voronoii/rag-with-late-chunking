import os, sys
import pickle
import torch
from transformers import AutoTokenizer
from utils import load_model, upsert_data_from_file
from embedder import LateChunkingEmbedder
import qdrant_client
from FlagEmbedding import FlagReranker
import numpy as np
import config as cfg
def setup():
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # os.system("docker run -d -p 6333:6333 -p 6334:6334 -v /data/talab/mj/qdrant_storage:/qdrant/storage:z qdrant/qdrant")

    client = qdrant_client.QdrantClient(
    host="localhost",
        port=6333, 
        timeout=15.0,
        
    )
    return client


def upsert_data(client, file_path: str, collection_name: str):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print("length of data: ", len(data))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model, has_instructions = load_model(cfg.embedding_model_name)
    embedding_tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model_name, trust_remote_code=True)
    lc = LateChunkingEmbedder(embedding_model, embedding_tokenizer, chunking_strategy="semantic", embedding_model_name=cfg.embedding_model_name)
    print("late chunking embedder loaded")

    lc.upload_to_qdrant(data, client, collection_name, batch_size=8)


def rag_with_bge(query, client, collection_name, k=5):
    reranker = FlagReranker('dragonkue/bge-reranker-v2-m3-ko', use_fp16=True)
    collection_name = "spatial2025_deduplicated"

    recall_at_k, mrr_at_k, ndcg_at_k = [], [], []
    K = 5

    for _, row in tqdm(df.iterrows(), total=len(df), desc="📊 Rerank 기반 평가"):
        q = row["question"]
        gt_id = row["ground_truth_doc_id"]
        r, m, n = evaluate_single_query_rerank(
            query=q,
            ground_truth_doc_id=gt_id,
            qdrant_client=qdrant_client,
            embed_model=Settings.embed_model,
            collection_name=collection_name,
            k=K
        )
        recall_at_k.append(r)
        mrr_at_k.append(m)
        ndcg_at_k.append(n)

    print(f"\n📌 Rerank 기준 평가 결과 @K={K}")
    print(f"🔸 Recall@{K}: {np.mean(recall_at_k):.4f}")
    print(f"🔸 MRR@{K}:    {np.mean(mrr_at_k):.4f}")
    print(f"🔸 nDCG@{K}:   {np.mean(ndcg_at_k):.4f}")


def main():
    collection_name = "spatial2025_ver3"
    qdrant_client = setup()
    # upsert_data_from_file(qdrant_client, './results_2025-04-21.pkl', collection_name)
    upsert_data(qdrant_client, './crawled_news_2025-04-22_20250413_20250421.pkl', collection_name)


if __name__ == "__main__":
    main()