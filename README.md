# RAG with Qdrant using Late Chunking and Reranker

**Qdrant**를 벡터 데이터베이스로 활용하여 RAG 시스템을 구현한 코드. **Jina AI**에서 발표한 전략인 **Late Chunking** 기법을 적용하였고, 한국어 성능 향상을 위해 **BGE 기반 Reranker**를 연동하여 평가를 수행함.

---

## 주요 내용

- **Qdrant 기반 벡터 검색**
- **Late Chunking**: 전체 문서를 임베딩한 후 chunking 처리
- **Reranker 연동**: `dragonkue/bge-reranker-v2-m3-ko` 모델 사용
- **RAG 평가 지표 구현**: Recall\@K, MRR\@K, nDCG\@K

---

## 구조 요약

### 1. 문서 처리 및 적재

- 문서 데이터를 불러와 **청크 분할**
- HuggingFace 임베딩 모델로 벡터화
- Qdrant에 각 청크를 `PointStruct`로 저장

### 2. 질문 생성 (Ground Truth 생성)

- 전체 문서를 기준으로 LLM을 이용해 **질문-정답 쌍** 생성
- 평가셋은 `question`, `ground_truth_doc_id`로 구성된 CSV 형태

### 3. 평가 (RAG Retrieval)

- 사용자 질문을 벡터화하여 Qdrant에서 Top-K 청크 검색
- 청크 ID에서 `doc_id` 추출하여 평가 지표 계산

### 4. Reranker 적용 (선택)

- 검색된 청크에 대해 `query-text` 쌍 생성
- `FlagEmbedding` 기반 Reranker로 점수 계산 및 재정렬
- 재정렬된 결과 기준으로 Recall / MRR / nDCG 재평가

---

## 평가 결과 (@K=5 기준)

| RAG 방식                   | Recall\@5 | MRR\@5 | nDCG\@5 |
| ------------------------ | --------- | ------ | ------- |
| Base RAG                 | 0.2736    | 0.1901 | 0.2110  |
| Late Chunking 적용         | 0.5472    | 0.3803 | 0.4219  |
| Late Chunking + Reranker | 0.5472    | 0.4814 | 0.4983  |


---

## 사용 모델 및 도구

- **LLM**: OpenAI `gpt-4o-mini` (질문 생성, 적절성 평가)
- **Embedding**: `dragonkue/snowflake-arctic-embed-l-v2.0-ko`
- **Reranker**: `dragonkue/bge-reranker-v2-m3-ko` (`FlagEmbedding` 활용)
- **Vector DB**: `Qdrant`
- **프레임워크**: `LlamaIndex`, `Transformers`,


---

## 참고

- [Jina AI: The Power of Late Chunking](https://github.com/jina-ai/late-chunking)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [FlagEmbedding: BGE reranker](https://github.com/FlagOpen/FlagEmbedding)
- [LlamaIndex RAG Eval Tools](https://docs.llamaindex.ai/en/stable/examples/eval/)

