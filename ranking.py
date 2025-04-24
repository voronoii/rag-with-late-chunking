from FlagEmbedding import FlagReranker
import numpy as np
from llama_index.core import Settings

def evaluate(reranked, reranked_doc_ids, ground_truth_doc_id):
    hit = ground_truth_doc_id in reranked_doc_ids
    recall = 1 if hit else 0

    try:
        rank = reranked_doc_ids.index(ground_truth_doc_id) + 1
        mrr = 1.0 / rank
    except ValueError:
        mrr = 0.0

    # 정답이 몇 번째에 있는가에 따라 점수를 다르게 부여
    dcg = sum([
        1 / np.log2(i + 2) if doc_id == ground_truth_doc_id else 0
        for i, doc_id in enumerate(reranked_doc_ids)
    ])
    idcg = 1.0  # 이상적인 DCG
    ndcg = dcg / idcg

    return recall, mrr, ndcg



def evaluate_single_query_rerank(reranker, query, ground_truth_doc_id, qdrant_client, embed_model, collection_name, k=5):
    # 1. 쿼리 임베딩
    query_vector = embed_model.get_text_embedding(query)

    # 2. Qdrant에서 Top-K 검색
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k,
        with_payload=True
    )

    if not results:
        return 0, 0.0, 0.0  # 검색 실패 시

    # 3. query-document pairs 생성
    chunks = [pt.payload["text"] for pt in results]
    doc_ids = [pt.id.split("~")[0] for pt in results]
    pairs = [[query, text] for text in chunks]

    # 4. reranker 점수 계산
    scores = reranker.compute_score(pairs, normalize=True)

    # 5. 점수 기준으로 재정렬
    reranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    reranked_doc_ids = [doc_id for doc_id, _ in reranked]

    return (reranked, reranked_doc_ids)

import qdrant_client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def hybrid_reranking(query, qdrant_client, collection_name, k=5):
    def arctan_normalize(score):
        return (2 / np.pi) * np.arctan(score)
    
    def search_and_rerank_hybrid(query, qdrant_client, collection_name, dense_model, reranker,
                              k=100, weights=None, decay=0.01):
        """
        3가지 스코어 를 모두 결합하여 rerank합니다. # + 날짜 점수는 제외
        weights: {"dense": 0.4, "sparse": 0.4, "bm25": 0.1, "recency": 0.1}
        """
        if weights is None:
            weights = {"dense": 0.4, "sparse": 0.4, "bm25": 0.2, } # "recency": 0.1

        # Step 1. 쿼리 임베딩
        dense_vector = dense_model.get_text_embedding(query)

        # Step 2. Qdrant에서 dense + sparse 검색
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=dense_vector,
            limit=k,
            with_payload=True
        )

        if not results:
            return []

        # Step 3. 개별 점수 분리
        doc_ids = [pt.id.split("~")[0] for pt in results]
        texts = [pt.payload["text"] for pt in results]
        # dates = [pt.payload.get("published_date", "1900-01-01") for pt in results]
        
        query_doc_pairs = [[query, text] for text in texts]

        dense_scores = [pt.score for pt in results]  # Qdrant cosine similarity
        sparse_scores = reranker.compute_score(query_doc_pairs, normalize=True)   
        bm25_scores = [pt.payload.get("bm25_score", 0) for pt in results]

        # bm25 점수 정규화
        bm25_scores = arctan_normalize(bm25_scores)
        
        # recency_scores = [get_recency_score(d, decay=decay) for d in dates]
        final_scores = [
            weights["dense"] * d + weights["sparse"] * s + weights["bm25"] * b
            for d, s, b in zip(dense_scores, sparse_scores, bm25_scores )
        ]
        
        reranked = sorted(zip(doc_ids, texts, final_scores), key=lambda x: x[2], reverse=True)

        return reranked  # [(doc_id, text, final_score), ...]



    sparse_reranker = FlagReranker("dragonkue/bge-reranker-v2-m3-ko", use_fp16=True)

    

    # vector search 결과 중 상위 100개 문서에 대해서만 처리 -> hybrid scoring 적용
    results = search_and_rerank_hybrid(
        query=query,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        dense_model=Settings.embed_model,
        reranker=sparse_reranker,
        k=100,
        weights={"dense": 0.3, "sparse": 0.5, "bm25": 0.2 }, # , "recency": 0.1
        decay=0.01
    )

    for i, (doc_id, text, score) in enumerate(results[:10], 1):
        print(f"🔹 Rank {i} | Score: {score:.4f}")
        print(f"{text}...\n")

    return results