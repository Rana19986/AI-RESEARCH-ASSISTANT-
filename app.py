import os
import pickle
import faiss
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="AI Research Paper Assistant")

BASE = os.path.dirname(os.path.abspath(__file__))

index = faiss.read_index(
    os.path.join(BASE, "research_index.faiss")
)

with open(os.path.join(BASE, "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)


class Question(BaseModel):
    question: str
    k: int = 5


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "AI Research Assistant is running!"
    }


@app.post("/ask")
def ask(data: Question):

    embedding = model.encode(
        [data.question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    k = min(data.k, len(chunks))

    scores, indices = index.search(
        embedding,
        k
    )

    results = []

    for rank, idx in enumerate(indices[0]):

        chunk = chunks[idx]

        results.append({
            "rank": rank + 1,
            "page": chunk.get("page"),
            "score": float(scores[0][rank]),
            "text": chunk.get("text", "")
        })

    return {
        "question": data.question,
        "results": results
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
