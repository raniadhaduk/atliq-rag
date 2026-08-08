"""
Step 6: Guardrails - out-of-scope detection + PII protection.

Run with:  python src\06_guardrails.py
"""

import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from presidio_analyzer import AnalyzerEngine

load_dotenv()

COLLECTION_NAME = "atliq_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
TOP_K = 5

# If the BEST match scores below this, we consider the question out of scope.
# Cosine similarity ranges roughly 0-1; tune this based on testing.
RELEVANCE_THRESHOLD = 0.35

ROLE_PERMISSIONS = {
    "hr_user":          ["hr", "general"],
    "finance_user":     ["finance", "general"],
    "marketing_user":   ["marketing", "general"],
    "engineering_user": ["engineering", "general"],
    "exec_user":        ["hr", "finance", "marketing", "engineering", "general"],
}

# Presidio's PII detector - loads pretrained NER models to spot
# emails, phone numbers, names, etc. in text.
pii_analyzer = AnalyzerEngine()


def check_pii(text):
    """
    Scan text for PII. Returns a list of detected entity types found,
    e.g. ["EMAIL_ADDRESS", "PERSON"]. Empty list = nothing detected.
    """
    results = pii_analyzer.analyze(text=text, language="en")
    return list({r.entity_type for r in results})


def retrieve_chunks(question, model, client, allowed_departments, top_k=TOP_K):
    query_vector = model.encode(question).tolist()
    rbac_filter = Filter(
        must=[FieldCondition(key="department", match=MatchAny(any=allowed_departments))]
    )
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=rbac_filter,
        limit=top_k,
    ).points
    return results


def build_prompt(question, retrieved_chunks):
    context_block = "\n\n---\n\n".join(
        f"[Source: {c.payload['filename']} | Department: {c.payload['department']}]\n"
        f"{c.payload['text']}"
        for c in retrieved_chunks
    )
    system_prompt = (
        "You are AtliQ Corp's internal assistant. Answer the user's question "
        "using ONLY the information in the CONTEXT below. "
        "If the context does not contain enough information to answer, say so clearly."
    )
    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"
    return system_prompt, user_prompt


def ask_question(question, role, model, qdrant_client, groq_client):
    allowed_departments = ROLE_PERMISSIONS.get(role)
    if allowed_departments is None:
        return f"Unknown role '{role}'.", [], []

    retrieved = retrieve_chunks(question, model, qdrant_client, allowed_departments)

    # --- GUARDRAIL 1: Out-of-scope detection ---
    top_score = retrieved[0].score if retrieved else 0.0
    if top_score < RELEVANCE_THRESHOLD:
        return (
            "I couldn't find relevant information in the documents I have access to "
            "to answer that question. This assistant only answers questions about "
            "AtliQ Corp's internal data.",
            retrieved,
            [],
        )

    # --- GUARDRAIL 2: PII detection on the retrieved context ---
    combined_text = " ".join(c.payload["text"] for c in retrieved)
    pii_found = check_pii(combined_text)

    system_prompt, user_prompt = build_prompt(question, retrieved)
    response = groq_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content

    return answer, retrieved, pii_found


if __name__ == "__main__":
    print("Loading models...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(path="qdrant_storage")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Ready.\n")

    print("=" * 60)
    print("AtliQ RAG Chatbot (RBAC + Guardrails)")
    print("Available roles:", ", ".join(ROLE_PERMISSIONS.keys()))
    print("=" * 60)

    role = input("\nLog in as role: ").strip()
    while role not in ROLE_PERMISSIONS:
        role = input(f"Unknown role. Choose from {list(ROLE_PERMISSIONS.keys())}: ").strip()

    print(f"\nLogged in as: {role}. Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer, retrieved, pii_found = ask_question(question, role, embed_model, qdrant_client, groq_client)

        print("\n--- Answer ---")
        print(answer)

        if retrieved:
            top_score = retrieved[0].score
            print(f"\n(Top relevance score: {top_score:.3f})")

        if pii_found:
            print(f"⚠️  PII detected in source context: {pii_found}")

        print()