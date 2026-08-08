"""
Step 3: Embeddings + Vector Store - convert each chunk's text into
a vector (a list of numbers capturing its meaning), and store it in
Qdrant so we can later search "which chunks are closest in meaning
to this question?"

Run with:  python src\03_embed_and_store.py
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from importlib import import_module
import sys
sys.path.append("src")
ingest = import_module("01_ingest_documents")
chunker = import_module("02_chunk_documents")

COLLECTION_NAME = "atliq_docs"
# This is a small, fast, free embedding model that runs locally -
# no API key, no internet needed after the first download.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def build_vector_store():
    # --- Step A: get our chunks (reusing steps 1 and 2) ---
    print("Loading and chunking documents...")
    documents = ingest.load_all_documents()
    chunks = chunker.chunk_documents(documents)
    print(f"Got {len(chunks)} chunks to embed")

    # --- Step B: load the embedding model ---
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(first run downloads the model, ~90MB, may take a minute)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # --- Step C: connect to Qdrant in local file mode ---
    # path="qdrant_storage" means Qdrant saves everything to a local
    # folder instead of needing a running server (Docker) elsewhere.
    client = QdrantClient(path="qdrant_storage")

    # --- Step D: create a "collection" (like a table) to hold our vectors ---
    # vector_size must match what our embedding model outputs.
    # all-MiniLM-L6-v2 produces 384-dimensional vectors.
    vector_size = model.get_sentence_embedding_dimension()

    if client.collection_exists(COLLECTION_NAME):
        print(f"\nCollection '{COLLECTION_NAME}' already exists - deleting to rebuild fresh")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}' (vector size: {vector_size})")

    # --- Step E: embed every chunk and upload to Qdrant ---
    print(f"\nEmbedding {len(chunks)} chunks...")
    texts = [chunk["text"] for chunk in chunks]

    # encode() can take a list of texts and embed them all at once - much
    # faster than looping and embedding one at a time.
    embeddings = model.encode(texts, show_progress_bar=True)

    # Build "points" - each point = one vector + its metadata (payload)
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,                              # unique ID per chunk
                vector=embedding.tolist(),
                payload={                          # payload = metadata stored alongside the vector
                    "text": chunk["text"],
                    "department": chunk["department"],
                    "filename": chunk["filename"],
                    "source_path": chunk["source_path"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
        )

    print("\nUploading to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Done. {len(points)} chunks stored in Qdrant.")

    return client, model


if __name__ == "__main__":
    client, model = build_vector_store()

    # --- Quick sanity test: search for something and see what comes back ---
    print("\n" + "=" * 60)
    print("Sanity check: searching for a test query")
    print("=" * 60)

    test_query = "What was the marketing budget?"
    query_vector = model.encode(test_query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,
    ).points

    print(f"\nQuery: '{test_query}'")
    print("Top 3 matches:\n")
    for r in results:
        print(f"  Score: {r.score:.4f} | [{r.payload['department']}] {r.payload['filename']}")
        print(f"  Text: {r.payload['text'][:150]}...")
        print()