"""
Step 4: Full RAG - given a user question, retrieve relevant chunks
from Qdrant, then ask Groq to answer using ONLY that retrieved context.

This is the first script where you can actually "chat" with your documents.

Run with:  python src\04_ask_question.py
"""

import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "atliq_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
TOP_K = 5  # how many chunks to retrieve per question


def retrieve_chunks(question, model, client, top_k=TOP_K):
    """
    Embed the question, then ask Qdrant for the top_k chunks
    whose vectors are closest in meaning to the question's vector.
    """
    query_vector = model.encode(question).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return results


def build_prompt(question, retrieved_chunks):
    """
    Stitch retrieved chunks into a single context block, then wrap
    it in instructions telling the LLM to answer ONLY from this context.
    This instruction is what stops the LLM from making things up
    using its own general knowledge instead of your private documents.
    """
    context_block = "\n\n---\n\n".join(
        f"[Source: {c.payload['filename']} | Department: {c.payload['department']}]\n"
        f"{c.payload['text']}"
        for c in retrieved_chunks
    )

    system_prompt = (
        "You are AtliQ Corp's internal assistant. Answer the user's question "
        "using ONLY the information in the CONTEXT below. "
        "If the context does not contain enough information to answer, "
        "say so clearly instead of guessing or using outside knowledge. "
        "When possible, mention which source document your answer came from."
    )

    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"

    return system_prompt, user_prompt


def ask_question(question, model, qdrant_client, groq_client):
    # Step 1: Retrieve
    retrieved = retrieve_chunks(question, model, qdrant_client)

    # Step 2: Build the prompt with retrieved context
    system_prompt, user_prompt = build_prompt(question, retrieved)

    # Step 3: Ask the LLM
    response = groq_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temperature - we want factual, grounded answers
    )

    answer = response.choices[0].message.content
    return answer, retrieved


if __name__ == "__main__":
    print("Loading embedding model and connecting to Qdrant + Groq...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(path="qdrant_storage")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Ready.\n")

    print("=" * 60)
    print("AtliQ RAG Chatbot (no RBAC yet - can see ALL documents)")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer, retrieved = ask_question(question, embed_model, qdrant_client, groq_client)

        print("\n--- Answer ---")
        print(answer)

        print("\n--- Sources used ---")
        for r in retrieved:
            print(f"  [{r.payload['department']}] {r.payload['filename']} (score: {r.score:.3f})")