import os
import pickle
import faiss
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(
    BASE_DIR,
    "research_index.faiss"
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "chunks.pkl"
)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Research Paper Assistant",
    description="RAG-based research paper semantic search API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL RESOURCES
# ============================================================

index = None
chunks = None
model = None


# ============================================================
# LOAD RAG RESOURCES
# ============================================================

@app.on_event("startup")
def load_resources():

    global index
    global chunks
    global model

    print("========================================")
    print("Starting AI Research Assistant")
    print("========================================")

    # -----------------------------
    # Load FAISS index
    # -----------------------------

    print("Loading FAISS index...")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    index = faiss.read_index(INDEX_PATH)

    print(
        f"FAISS loaded successfully | "
        f"Vectors: {index.ntotal} | "
        f"Dimensions: {index.d}"
    )

    # -----------------------------
    # Load chunks
    # -----------------------------

    print("Loading chunks...")

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    print(
        f"Chunks loaded successfully | "
        f"Total chunks: {len(chunks)}"
    )

    # -----------------------------
    # Check index/chunk consistency
    # -----------------------------

    if index.ntotal != len(chunks):

        raise ValueError(
            f"FAISS vectors ({index.ntotal}) "
            f"do not match chunks ({len(chunks)})"
        )

    # -----------------------------
    # Load embedding model
    # -----------------------------

    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(MODEL_NAME)

    model_dimension = model.get_sentence_embedding_dimension()

    print(
        f"Embedding dimension: {model_dimension}"
    )

    # -----------------------------
    # Check embedding dimension
    # -----------------------------

    if model_dimension != index.d:

        raise ValueError(
            f"Embedding dimension mismatch! "
            f"FAISS index = {index.d}, "
            f"model = {model_dimension}"
        )

    print("========================================")
    print("AI Research Assistant is ready!")
    print("========================================")


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):

    question: str

    k: int = 5


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "AI Research Assistant is running!",
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "chunks_loaded": chunks is not None,
        "model_loaded": model is not None,
        "vectors": index.ntotal if index else 0,
        "chunks": len(chunks) if chunks else 0,
        "embedding_dimension": (
            model.get_sentence_embedding_dimension()
            if model else None
        )
    }


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_context(question: str, k: int = 5):

    if model is None:
        raise RuntimeError(
            "Embedding model is not loaded."
        )

    if index is None:
        raise RuntimeError(
            "FAISS index is not loaded."
        )

    if chunks is None:
        raise RuntimeError(
            "Chunks are not loaded."
        )

    # Keep k within valid range
    k = max(
        1,
        min(k, index.ntotal)
    )

    # ----------------------------------------
    # Convert question into embedding
    # ----------------------------------------

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32
    )

    # ----------------------------------------
    # Search FAISS
    # ----------------------------------------

    scores, indices = index.search(
        question_embedding,
        k
    )

    results = []

    # ----------------------------------------
    # Process results
    # ----------------------------------------

    for rank, idx in enumerate(indices[0]):

        if idx < 0:
            continue

        if idx >= len(chunks):
            continue

        chunk = chunks[idx]

        # Handle dictionary chunks
        if isinstance(chunk, dict):

            text = chunk.get(
                "text",
                ""
            )

            page = chunk.get(
                "page",
                None
            )

        # Handle string chunks
        else:

            text = str(chunk)
            page = None

        results.append({
            "rank": rank + 1,
            "page": page,
            "score": float(
                scores[0][rank]
            ),
            "text": text
        })

    return results


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    # ----------------------------------------
    # Clean question
    # ----------------------------------------

    question = request.question.strip()

    # ----------------------------------------
    # Validate question
    # ----------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if len(question) > 2000:

        raise HTTPException(
            status_code=400,
            detail=(
                "Question is too long. "
                "Maximum 2000 characters."
            )
        )

    try:

        # ------------------------------------
        # Retrieve relevant chunks
        # ------------------------------------

        results = retrieve_context(
            question=question,
            k=request.k
        )

        # ------------------------------------
        # No results
        # ------------------------------------

        if not results:

            return {
                "question": question,
                "answer": (
                    "No relevant information "
                    "was found in the research paper."
                ),
                "results": []
            }

        # ------------------------------------
        # Build context
        # ------------------------------------

        context_parts = []

        for result in results:

            page = result["page"]
            text = result["text"]

            context_parts.append(
                f"Page {page}:\n{text}"
            )

        context = "\n\n".join(
            context_parts
        )

        # ------------------------------------
        # Return retrieval results
        # ------------------------------------

        return {
            "question": question,

            "answer": (
                "Relevant information retrieved "
                "from the research paper."
            ),

            "context": context,

            "results": results
        }

    except Exception as error:

        print(
            f"ERROR: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process the question."
            )
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
        )
