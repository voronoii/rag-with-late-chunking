from FlagEmbedding import FlagReranker
import numpy as np


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
