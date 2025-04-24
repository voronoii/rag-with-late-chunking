import os
import torch
from transformers import AutoModel
from tqdm import tqdm
import uuid
from qdrant_client.models import PointStruct
import pickle

def remove_unsupported_kwargs(original_encode):
    def wrapper(self, *args, **kwargs):
        # Remove 'prompt_name' from kwargs if present
        kwargs.pop('prompt_name', None)
        kwargs.pop('request_qid', None)
        return original_encode(self, *args, **kwargs)

    return wrapper

def load_model(model_name, model_weights=None, **model_kwargs):
    MODELS_WITHOUT_PROMPT_NAME_ARG = [
    'jinaai/jina-embeddings-v2-small-en',
    'jinaai/jina-embeddings-v2-base-en',
    'jinaai/jina-embeddings-v3',
]
    
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    has_instructions = False

    if model_weights and os.path.exists(model_weights):
        model._model.load_state_dict(torch.load(model_weights, device=model.device))

    # encode functions of various models do not support all sentence transformers kwargs parameter
    if model_name in MODELS_WITHOUT_PROMPT_NAME_ARG:
        ENCODE_FUNC_NAMES = ['encode', 'encode_queries', 'encode_corpus']
        for func_name in ENCODE_FUNC_NAMES:
            if hasattr(model, func_name):
                setattr(
                    model,
                    func_name,
                    remove_unsupported_kwargs(getattr(model, func_name)),
                )

    return model, has_instructions


def upsert_data_from_file(qdrant_client, file_path, collection_name: str):

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print("length of data: ", len(data))

    BATCH_SIZE = 50 

    for i in tqdm(range(0, len(data), BATCH_SIZE)): # tqdm 사용하기 위해 추가
        batch = data[i:i + BATCH_SIZE]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=doc["embedding"].tolist(),
                payload={
                    "text": doc["text"],
                    "doc_id": doc["doc_id"],
                    "chunk_index": doc["chunk_index"]
                }
            )
            for doc in batch
        ]

        qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)

    print(f"✅ Uploaded {len(points)} points to Qdrant collection '{collection_name}'")


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
import qdrant_client
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm




def add_bm25_scores(qdrant_client, collection_name: str):
    batch_size = 512

    # 1. 전체 문서 및 ID 수집
    offset = None
    all_docs = []
    all_ids = []

    while True:
        docs, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            offset=offset,
            limit=batch_size
        )
        if not docs:
            break

        for pt in docs:
            all_docs.append(pt.payload["text"])
            all_ids.append(pt.id)  # pt.id는 str 또는 int

        if next_offset is None:
            break
        offset = next_offset

    # 2. BM25 점수 계산 (TF-IDF 총합으로 근사)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    bm25_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # 3. Qdrant payload에 저장
    for pt_id, score in tqdm(zip(all_ids, bm25_scores), total=len(all_ids), desc="🔄 Updating BM25 scores"):
        qdrant_client.set_payload(
            collection_name=collection_name,
            payload={"bm25_score": float(score)},
            points=[pt_id]  # ✅ PointId 사용하지 않고 그대로 전달
        )

    print(f"✅ BM25-like 점수 {len(all_ids)}건 저장 완료")
