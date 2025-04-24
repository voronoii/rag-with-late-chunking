from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import numpy as np
from pooling import chunked_pooling
from chunking import Chunker
from typing import List, Tuple, Union, Dict
from tqdm import tqdm
import requests
import os
import sys
import time
from qdrant_client.models import PointStruct, VectorParams, Distance, OptimizersConfigDiff
import pickle
from datetime import datetime
import uuid
import hashlib
class LateChunkingEmbedder:

    def __init__(self, 
            model: AutoModel,
            tokenizer: AutoTokenizer, 
            chunking_strategy: str = "sentences",
            n_sentences: int = 1,
            embedding_model_name: str = None,
            device: str = "cuda",
            max_length: int = 2000
        ):
        self.max_length = max_length
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.chunker = Chunker(chunking_strategy = chunking_strategy, embedding_model_name=embedding_model_name)
        self.n_sentences = n_sentences

    
    # 단일 문서를 테스트할 때만 사용
    def run(self, document: str):
        annotations = [self.chunker.chunk(text=document, tokenizer=self.tokenizer, n_sentences=self.n_sentences)]
        model_inputs = self.tokenizer(
            document,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
                       
        # Forward pass
        with torch.no_grad():
            model_outputs = self.model(**model_inputs)
        
        self.output_embs = chunked_pooling(
            model_outputs, annotations, max_length=self.max_length, 
        )[0]
        return self.output_embs

    

    '''
    qdrant에 적재 시 distance를 cosine으로 하는 경우 embedding normalization 필요하나 qdrant에서 자동으로 처리됨
    '''
    def upload_to_qdrant(
        self,
        documents: List[str],
        qdrant_client,
        collection_name: str,
        batch_size: int = 8
    ):  
        results = self.batch_embed_with_chunks(documents, batch_size=batch_size)

        if not qdrant_client.collection_exists(collection_name):
            print("embedding dim : ", results[0]["embedding"].shape[0])
            qdrant_client.create_collection(collection_name=collection_name, 
                                            vectors_config=VectorParams(size=results[0]["embedding"].shape[0]
                                                                        , distance=Distance.COSINE
                                                                        , on_disk=True),
                                            optimizers_config=OptimizersConfigDiff(indexing_threshold=100, 
                                                                            )
                                                                                )
                                                                                                           
                                                                                                                                                                          
            print(f"✅ Created collection : '{collection_name}'/n")
        
        

        BATCH_SIZE = 50  # 또는 50개 정도로 줄이기

        for i in tqdm(range(0, len(results), BATCH_SIZE)): # tqdm 사용하기 위해 추가
            batch = results[i:i + BATCH_SIZE]

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=doc["embedding"].tolist(),
                    payload={
                        "text": doc["text"],
                        "doc_id": doc["doc_id"],
                        "chunk_index": doc["chunk_index"],
                        "published_date" : doc["published_date"]
                    }
                )
                for doc in batch
            ]

            qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)

        print(f"✅ Uploaded {len(points)} points to Qdrant collection '{collection_name}'")

    
    def deduplicate_chunks_by_text(self, chunks): # chunks : List[Dict[str, Union[np.ndarray, str, int]]]
        print("🔍 Removing duplicate chunks...")
        print("Original chunks : ", len(chunks))
        seen_hashes = set()
        deduped = []
        for c in chunks:
            h = hashlib.md5(c["text"].strip().encode()).hexdigest()
            if h not in seen_hashes:
                deduped.append(c)
                seen_hashes.add(h)
        print("Deduplicated chunks : ", len(deduped), "/n")
        return deduped    
    
    
    '''
    모델이 이미 GPU에 올라간 경우는, 배치 처리 방식이 가장 효율적이고 안정적.
    ProcessPoolExecutor는 GPU 사용 불가한 CPU 전용 작업에는 좋지만, 
    GPU 사용 가능한 작업에는 속도가 느리므로 사용하지 않음.
    '''
    def batch_embed_with_chunks(
        self,
        documents: List[str],
        batch_size: int = 8
    ) -> List[Dict[str, Union[np.ndarray, str, int]]]:
        """
        문서를 배치로 임베딩하고 각 청크의 텍스트와 함께 반환

        Returns: List of dicts:
            {
                "embedding": np.ndarray,
                "text": str,
                "doc_id": int,
                "chunk_index": int
            }
        """
        results = []

        for i in tqdm(range(0, len(documents), batch_size), desc="Batch embedding"):
            batch_docs = documents[i:i + batch_size]
            batch_docs_only_contents = [f'{doc["title"]} {doc["content"]}' for doc in batch_docs]
            

            # 1. 청크 어노테이션
            # doc : dict
            annotations_batch = [
                self.chunker.chunk(text=f'{doc["title"]} {doc["content"]}', tokenizer=self.tokenizer, n_sentences=self.n_sentences)
                for doc in batch_docs
            ]

            # 2. 모델 인퍼런스를 위한 tokenizer
            model_inputs = self.tokenizer(
                batch_docs_only_contents,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

            # 3. 모델 인퍼런스
            with torch.no_grad():
                model_outputs = self.model(**model_inputs)

            # 4. 청크당 임베딩 계산
            batch_embs = chunked_pooling(model_outputs, annotations_batch, max_length=self.max_length)

            # 5. 각 청크 텍스트 복원 및 결과 결합
            for doc_idx, (doc, annotations, chunk_embs) in enumerate(zip(batch_docs, annotations_batch, batch_embs)):
                full_text = f'{doc["title"]} {doc["content"]}'
                doc_inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=self.max_length)
                input_ids = doc_inputs["input_ids"][0]  # shape: (seq_len,)

                for chunk_idx, ((start_idx, end_idx), emb) in enumerate(zip(annotations, chunk_embs)):
                    chunk_tokens = input_ids[start_idx:end_idx]
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                    results.append({
                        "embedding": emb,
                        "text": chunk_text,
                        "doc_id": i + doc_idx,
                        "chunk_index": chunk_idx,
                        "published_date": doc["publishDateTime"]
                    })

        # 중복 제거 
        deduped_results = self.deduplicate_chunks_by_text(results)
        
        # 에러 대비 결과저장
        today = datetime.now().strftime("%Y-%m-%d")
        with open(f'./results_{today}.pkl', 'wb') as f:
            pickle.dump(deduped_results, f)
        return deduped_results

  
   


    def query(self, query: str):
        if "output_embs" not in dir(self):
            raise ValueError("no embeddings calculated, use .run(document) to create chunk embeddings")
        
        model_inputs = self.tokenizer(
            query,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            query_embedding = self.model(**model_inputs)

        query_embedding = query_embedding.last_hidden_state[:, 0, :].detach().cpu().numpy()  
        
        similarities = []
        for emb in self.output_embs:
            emb = emb.reshape(1, -1)
            cos_sim = cosine_similarity(emb, query_embedding)
            similarities.append(cos_sim)
        
        return similarities

def cosine_similarity(vector1, vector2):
    vector1_norm = vector1 / np.linalg.norm(vector1)
    vector2_norm = vector2 / np.linalg.norm(vector2)
    # cosine 유사도 계산
    return np.sum(vector1_norm * vector2_norm, axis=-1)