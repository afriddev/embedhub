# Fast Embed & Rerank Service — RAG-ready

Blazing fast **embedding** and **cross-encoder reranker** service optimized for **Retrieval-Augmented Generation (RAG)**, **KG‑RAG (Knowledge-Graph RAG)**, and **Graph‑RAG** workflows. Self-host on a small GPU


## Minimal integration flow

1. `POST /api/v1/embedding` → get embeddings for documents/nodes.
2. Index embeddings into a vector store (FAISS / postgresql(pgvector) / etc.).
3. On user query: retrieve candidates from vector store.
4. `POST /api/v1/ce/reranker` with `query` + `documents` → get final top‑N.
5. Send top‑N to your LLM as context for generation.

## API (examples)

### Embeddings

**POST** `/api/v1/embedding`  
Request:

```json
{ "texts": ["doc1 text", "doc2 text"] }
```

Response:

```json
{ "embeddings": [[...],[...]], "dimensions": 1024 }
```

### Rerank

**POST** `/api/v1/ce/reranker`  
Request:

```json
{
  "query": "Find causes of X",
  "documents": ["candidate A", "candidate B", "..."],
  "returnDocuments": false,
  "topN": 10
}
```

Response:

```json

{
  "results":[
    {
    "docIndex"::17,
    "doctext":"",
    "score":0.99951171875
    }, ...

  ],
  "query":"Find causes of X"

}

```
## Environment (example `.env`)
```env
PORT = 8000
MAX_TOKEN_LIMIT_PER_TEXT = 500
EMBEDDING_MODEL_NAME = thenlper/gte-large
MAX_EMBEDDING_TEXTS_PER_REQUEST = 100
MAX_EMBEDDING_BATCH_REQUEST_DELAY = 5
MAX_EMBEDDING_BATCH_SIZE = 50
CROSS_ENCODER_MODEL_NAME = cross-encoder/ms-marco-MiniLM-L6-v2
MAX_CE_RE_RANKER_PAIRS = 200
MAX_CE_RE_RANKER_BACTH_SIZE = 100
MAX_CE_RE_RANKER_BACTH_REQUEST_DELAY = 5
```

## Performance (benchmarked)

- Embeddings: 20×100 tokens ≈ 200ms; 100×400 tokens ≈ 700ms
- Reranker: 100 docs × 300 tokens ≲ 300ms
- Throughput: 100 req/sec, 6000 req/min (observed)

## Deployment

```bash
git clone https://github.com/railtelai/embedhub.git
cd embedhub
pip install -r requirements.txt
python main.py
```

