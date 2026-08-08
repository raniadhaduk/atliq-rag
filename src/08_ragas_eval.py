"""
Step 7b: Custom RAG Evaluation - implements the same core idea as Ragas
(LLM-as-judge scoring) without the dependency headaches.
"""

import os
import json
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

ROLE_PERMISSIONS = {
    "exec_user": ["hr", "finance", "marketing", "engineering", "general"],
}

TEST_SET = [
    {
        "question": "How many employees are listed in the HR data?",
        "ground_truth": "There are 100 employees listed in the HR data.",
    },
    {
        "question": "What was the total marketing budget in 2024?",
        "ground_truth": "The total marketing budget in 2024 was $15M.",
    },
    {
        "question": "What columns does the HR employee data contain?",
        "ground_truth": (
            "The HR data contains employee_id, full_name, role, department, "
            "email, location, date_of_birth, date_of_joining, manager_id, "
            "salary, leave_balance, leaves_taken, attendance_pct, "
            "performance_rating, and last_review_date."
        ),
    },
    {
        "question": "What was the marketing spend in Q4 2024?",
        "ground_truth": "Marketing spend in Q4 2024 was $2.5 million.",
    },
]


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
    return [r.payload["text"] for r in results]


def build_prompt(question, retrieved_texts):
    context_block = "\n\n---\n\n".join(retrieved_texts)
    system_prompt = (
        "You are AtliQ Corp's internal assistant. Answer the user's question "
        "using ONLY the information in the CONTEXT below. Be concise and direct."
    )
    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"
    return system_prompt, user_prompt


def generate_answer(question, model, qdrant_client, groq_client, role="exec_user"):
    allowed = ROLE_PERMISSIONS[role]
    retrieved_texts = retrieve_chunks(question, model, qdrant_client, allowed)
    system_prompt, user_prompt = build_prompt(question, retrieved_texts)

    response = groq_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content
    return answer, retrieved_texts


def ask_judge_for_score(groq_client, grading_prompt):
    response = groq_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluation grader. Respond with ONLY a JSON "
                    'object like {"score": 0.0-1.0, "reason": "short explanation"}. '
                    "No other text, no markdown formatting."
                ),
            },
            {"role": "user", "content": grading_prompt},
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        return float(parsed["score"]), parsed.get("reason", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        return None, f"Could not parse judge response: {raw}"


def score_faithfulness(groq_client, question, answer, contexts):
    context_block = "\n\n".join(contexts)
    prompt = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        "Grade how well the ANSWER is supported by the CONTEXT alone. "
        "A score of 1.0 means every claim in the answer is directly backed by "
        "the context. A score of 0.0 means the answer contains information "
        "not found in the context (hallucination). Penalize any unsupported claims."
    )
    return ask_judge_for_score(groq_client, prompt)


def score_answer_relevance(groq_client, question, answer):
    prompt = (
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        "Grade how directly the ANSWER addresses the QUESTION asked. "
        "1.0 = fully relevant and on-topic. 0.0 = does not address the "
        "question at all, or answers a different question."
    )
    return ask_judge_for_score(groq_client, prompt)


def score_context_precision(groq_client, question, contexts, ground_truth):
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i+1}]: {c}" for i, c in enumerate(contexts)
    )
    prompt = (
        f"QUESTION: {question}\n\n"
        f"GROUND TRUTH ANSWER: {ground_truth}\n\n"
        f"RETRIEVED CHUNKS:\n{context_block}\n\n"
        "Grade what proportion of the retrieved chunks are actually useful "
        "and relevant for answering the question, given the ground truth "
        "answer. 1.0 = all chunks are relevant and useful. 0.0 = none of "
        "the chunks are relevant."
    )
    return ask_judge_for_score(groq_client, prompt)


if __name__ == "__main__":
    print("Loading models...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(path="qdrant_storage")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    print("Running pipeline + evaluation on test set...\n")

    all_results = []

    for case in TEST_SET:
        question = case["question"]
        ground_truth = case["ground_truth"]

        print(f"Question: {question}")
        answer, contexts = generate_answer(question, embed_model, qdrant_client, groq_client)
        print(f"  Answer: {answer[:100]}...")

        faith_score, faith_reason = score_faithfulness(groq_client, question, answer, contexts)
        rel_score, rel_reason = score_answer_relevance(groq_client, question, answer)
        prec_score, prec_reason = score_context_precision(groq_client, question, contexts, ground_truth)

        print(f"  Faithfulness:      {faith_score}")
        print(f"  Answer Relevance:  {rel_score}")
        print(f"  Context Precision: {prec_score}")
        print()

        all_results.append({
            "question": question,
            "answer": answer,
            "faithfulness": faith_score,
            "answer_relevance": rel_score,
            "context_precision": prec_score,
        })

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    valid_faith = [r["faithfulness"] for r in all_results if r["faithfulness"] is not None]
    valid_rel = [r["answer_relevance"] for r in all_results if r["answer_relevance"] is not None]
    valid_prec = [r["context_precision"] for r in all_results if r["context_precision"] is not None]

    if valid_faith:
        print(f"Average Faithfulness:      {sum(valid_faith)/len(valid_faith):.2f}")
    if valid_rel:
        print(f"Average Answer Relevance:  {sum(valid_rel)/len(valid_rel):.2f}")
    if valid_prec:
        print(f"Average Context Precision: {sum(valid_prec)/len(valid_prec):.2f}")

    print("\nPer-question breakdown:")
    for r in all_results:
        print(f"  [{r['question'][:50]}...] "
              f"faith={r['faithfulness']} rel={r['answer_relevance']} prec={r['context_precision']}")