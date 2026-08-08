"""
Step 7: Evaluation & Monitoring - add LangSmith tracing to see exactly
what happens inside each RAG call, plus basic Ragas evaluation metrics.

Run with:  python src\07_evaluate_and_trace.py
"""

import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langsmith import traceable

load_dotenv()

COLLECTION_NAME = "atliq_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.35

ROLE_PERMISSIONS = {
    "hr_user":          ["hr", "general"],
    "finance_user":     ["finance", "general"],
    "marketing_user":   ["marketing", "general"],
    "engineering_user": ["engineering", "general"],
    "exec_user":        ["hr", "finance", "marketing", "engineering", "general"],
}


# --- Each @traceable function becomes its own visible "step" in LangSmith ---

@traceable(name="retrieve_chunks")
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
    # Return plain dicts so LangSmith can log them cleanly (not custom objects)
    return [
        {"text": r.payload["text"], "department": r.payload["department"],
         "filename": r.payload["filename"], "score": r.score}
        for r in results
    ]


@traceable(name="build_prompt")
def build_prompt(question, retrieved_chunks):
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['filename']} | Department: {c['department']}]\n{c['text']}"
        for c in retrieved_chunks
    )
    system_prompt = (
        "You are AtliQ Corp's internal assistant. Answer the user's question "
        "using ONLY the information in the CONTEXT below. "
        "If the context does not contain enough information to answer, say so clearly."
    )
    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"
    return system_prompt, user_prompt


@traceable(name="call_llm", run_type="llm")
def call_llm(system_prompt, user_prompt, groq_client):
    response = groq_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


@traceable(name="ask_question_full_pipeline")
def ask_question(question, role, model, qdrant_client, groq_client):
    """
    This top-level function ties everything together. Because it's
    @traceable, LangSmith will show it as a parent "run" containing
    all the nested retrieve_chunks / build_prompt / call_llm steps -
    giving you a full waterfall view of one request.
    """
    allowed_departments = ROLE_PERMISSIONS.get(role)
    if allowed_departments is None:
        return f"Unknown role '{role}'."

    retrieved = retrieve_chunks(question, model, qdrant_client, allowed_departments)

    top_score = retrieved[0]["score"] if retrieved else 0.0
    if top_score < RELEVANCE_THRESHOLD:
        return (
            "I couldn't find relevant information in the documents I have access "
            "to answer that question."
        )

    system_prompt, user_prompt = build_prompt(question, retrieved)
    answer = call_llm(system_prompt, user_prompt, groq_client)
    return answer


if __name__ == "__main__":
    print("Loading models...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(path="qdrant_storage")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Ready. LangSmith tracing is active if LANGSMITH_TRACING=true in .env\n")

    role = input("Log in as role: ").strip()
    while role not in ROLE_PERMISSIONS:
        role = input(f"Unknown role. Choose from {list(ROLE_PERMISSIONS.keys())}: ").strip()

    print(f"\nLogged in as: {role}. Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = ask_question(question, role, embed_model, qdrant_client, groq_client)

        print("\n--- Answer ---")
        print(answer)
        print("\n(Check smith.langchain.com to see the full trace for this question)\n")