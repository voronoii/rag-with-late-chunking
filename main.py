import os, sys
import pickle
import torch
from transformers import AutoTokenizer
from utils import load_model, upsert_data_from_file
from embedder import LateChunkingEmbedder
import qdrant_client
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

    model_name = 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model, has_instructions = load_model(model_name)
    embedding_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lc = LateChunkingEmbedder(embedding_model, embedding_tokenizer, chunking_strategy="semantic", embedding_model_name=model_name)
    print("late chunking embedder loaded")

    lc.upload_to_qdrant(data, client, collection_name, batch_size=8)



def main():
    collection_name = "spatial2025_deduplicated"
    qdrant_client = setup()
    upsert_data_from_file(qdrant_client, './results_2025-04-21.pkl', collection_name)


if __name__ == "__main__":
    main()