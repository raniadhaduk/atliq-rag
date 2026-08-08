"""
Step 9: Streamlit frontend - wraps our full RAG + RBAC + guardrails
pipeline in a real web chat interface.

Run with:  streamlit run src\10_app.py
"""

import os
import streamlit as st
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
RELEVANCE_THRESHOLD = 0.35

ROLE_PERMISSIONS = {
    "hr_user":          ["hr", "general"],
    "finance_user":     ["finance", "general"],
    "marketing_user":   ["marketing", "general"],
    "engineering_user": ["engineering", "general"],
    "exec_user":        ["hr", "finance", "marketing", "engineering", "general"],
}


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_qdrant_client():
    return QdrantClient(path="qdrant_storage")


@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


@st.cache_resource
def load_pii_analyzer():
    return AnalyzerEngine()


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


def check_pii(analyzer, text):
    results = analyzer.analyze(text=text, language="en")
    return list({r.entity_type for r in results})


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


GREETINGS = {
    "hi", "hello", "hey", "hii", "hiya",
    "good morning", "good afternoon", "good evening",
}


def ask_question(question, role, model, qdrant_client, groq_client, pii_analyzer):
    # Handle small talk without going through retrieval/guardrails at all -
    # no point running semantic search over business documents for "hello".
    normalized = question.strip().lower().rstrip("!.?")
    if normalized in GREETINGS:
        return {
            "answer": (
                "Hello! I'm AtliQ Corp's internal assistant. "
                "Ask me anything about company data within your access level."
            ),
            "sources": [],
            "pii_found": [],
        }

    allowed_departments = ROLE_PERMISSIONS[role]
    retrieved = retrieve_chunks(question, model, qdrant_client, allowed_departments)

    top_score = retrieved[0].score if retrieved else 0.0
    if top_score < RELEVANCE_THRESHOLD:
        return {
            "answer": (
                "I couldn't find relevant information in the documents I have "
                "access to for that question."
            ),
            "sources": [],
            "pii_found": [],
        }

    combined_text = " ".join(c.payload["text"] for c in retrieved)
    pii_found = check_pii(pii_analyzer, combined_text)

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

    return {
        "answer": answer,
        "sources": retrieved,
        "pii_found": pii_found,
    }


# ================== STREAMLIT UI ==================

st.set_page_config(page_title="AtliQ RAG Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AtliQ Corp Internal Assistant")
st.caption("RAG-powered chatbot with Role-Based Access Control and Guardrails")

with st.sidebar:
    st.header("Login")
    role = st.selectbox(
        "Select your role:",
        options=list(ROLE_PERMISSIONS.keys()),
        format_func=lambda r: r.replace("_", " ").title(),
    )
    st.info(f"**Access level:** {', '.join(ROLE_PERMISSIONS[role])}")

    st.divider()
    st.caption(
        "Try asking about topics inside your access level, then switch roles "
        "and ask again to see RBAC in action."
    )

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

embed_model = load_embedding_model()
qdrant_client = load_qdrant_client()
groq_client = load_groq_client()
pii_analyzer = load_pii_analyzer()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.write(f"[{s['department']}] {s['filename']} (score: {s['score']:.3f})")
        if msg.get("pii_found"):
            st.warning(f"⚠️ PII detected in source context: {', '.join(msg['pii_found'])}")

question = st.chat_input("Ask a question about AtliQ Corp's internal data...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(question, role, embed_model, qdrant_client, groq_client, pii_analyzer)

        st.markdown(result["answer"])

        sources_display = []
        if result["sources"]:
            with st.expander("Sources used"):
                for s in result["sources"]:
                    st.write(f"[{s.payload['department']}] {s.payload['filename']} (score: {s.score:.3f})")
                    sources_display.append({
                        "department": s.payload["department"],
                        "filename": s.payload["filename"],
                        "score": s.score,
                    })

        if result["pii_found"]:
            st.warning(f"⚠️ PII detected in source context: {', '.join(result['pii_found'])}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources_display,
        "pii_found": result["pii_found"],
    })