"""
Step 5: RBAC - restrict retrieval based on the logged-in user's role.

The key idea: filtering happens INSIDE the Qdrant query itself, using
the "department" metadata we stored back in Step 3. If a chunk's
department isn't in the user's allowed list, Qdrant never returns it -
so the LLM never even sees it, let alone leaks it.

Run with:  python src\05_rbac_chat.py
"""

import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

load_dotenv()

COLLECTION_NAME = "atliq_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
TOP_K = 5

# --- Role -> allowed departments mapping ---
# "general" is included everywhere since it's low-sensitivity company-wide info.
ROLE_PERMISSIONS = {
    "hr_user":         ["hr", "general"],
    "finance_user":    ["finance", "general"],
    "marketing_user":  ["marketing", "general"],
    "engineering_user":["engineering", "general"],
    "exec_user":       ["hr", "finance", "marketing", "engineering", "general"],  # sees everything
}


def retrieve_chunks(question, model, client, allowed_departments, top_k=TOP_K):
    """
    Same as before, but now with a metadata filter: Qdrant will ONLY
    search among chunks whose 'department' field is in allowed_departments.
    """
    query_vector = model.encode(question).tolist()

    # This filter is the actual RBAC enforcement mechanism.
    rbac_filter = Filter(
        must=[
            FieldCondition(
                key="department",
                match=MatchAny(any=allowed_departments),
            )
        ]
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=rbac_filter,   # <-- this line enforces access control
        limit=top_k,
    ).points

    return results


def build_prompt(question, retrieved_chunks):
    if not retrieved_chunks:
        # Nothing matched within the user's allowed departments at all
        context_block = "(no relevant documents found within your access level)"
    else:
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
        "Never claim information exists outside of what's shown in the context."
    )

    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"

    return system_prompt, user_prompt


def ask_question(question, role, model, qdrant_client, groq_client):
    allowed_departments = ROLE_PERMISSIONS.get(role)

    if allowed_departments is None:
        return f"Unknown role '{role}'. Cannot process request.", []

    retrieved = retrieve_chunks(question, model, qdrant_client, allowed_departments)
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
    return answer, retrieved


if __name__ == "__main__":
    print("Loading embedding model and connecting to Qdrant + Groq...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(path="qdrant_storage")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Ready.\n")

    print("=" * 60)
    print("AtliQ RAG Chatbot (WITH RBAC)")
    print("Available roles:", ", ".join(ROLE_PERMISSIONS.keys()))
    print("=" * 60)

    role = input("\nLog in as role: ").strip()
    while role not in ROLE_PERMISSIONS:
        print(f"Unknown role. Choose from: {', '.join(ROLE_PERMISSIONS.keys())}")
        role = input("Log in as role: ").strip()

    print(f"\nLogged in as: {role}")
    print(f"You can access: {ROLE_PERMISSIONS[role]}")
    print("Type 'exit' to quit\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer, retrieved = ask_question(question, role, embed_model, qdrant_client, groq_client)

        print("\n--- Answer ---")
        print(answer)

        if retrieved:
            print("\n--- Sources used (within your access level) ---")
            for r in retrieved:
                print(f"  [{r.payload['department']}] {r.payload['filename']} (score: {r.score:.3f})")
        print()