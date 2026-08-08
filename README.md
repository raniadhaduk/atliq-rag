\# AtliQ RAG — Enterprise RAG with RBAC, Guardrails, and Evaluation



A production-style Retrieval Augmented Generation (RAG) chatbot built from

scratch to understand every layer of a real internal AI assistant: document

ingestion, chunking, vector search, role-based access control, guardrails,

observability, and automated evaluation.



Built as a hands-on learning project following the "RAG with RBAC, Guardrails

and Monitoring" brief from Codebasics' \*5 AI Projects That Will Matter in 2026\*.



\## Demo



A Streamlit chat interface lets you log in as different company roles

(`hr\_user`, `finance\_user`, `marketing\_user`, `engineering\_user`, `exec\_user`)

and ask questions about internal company documents. Access is enforced at the

retrieval layer — not just prompted for — so a marketing employee genuinely

cannot retrieve HR salary data, regardless of how the question is phrased.



```bash

streamlit run src/10\_app.py

```



\## Architecture



Documents (.md / .csv / .xlsx)

│

▼

Ingestion (Docling + pandas) ──► tags every doc with its department

│

▼

Chunking (LangChain RecursiveCharacterTextSplitter)

│

▼

Embeddings (sentence-transformers, all-MiniLM-L6-v2)

│

▼

Vector Store (Qdrant, local mode)

│

▼

User question ──► Embed ──► RBAC-filtered vector search ──► Relevance guardrail

│ │

│ (below threshold?

│ refuse, no LLM call)

▼

PII scan on retrieved context (Presidio)

│

▼

LLM generation (Groq / Llama 3.3 70B) — grounded strictly in retrieved context

│

▼

Answer + cited sources + PII warnings





Every request is traced end-to-end in \*\*LangSmith\*\*, and answer quality is

scored automatically with a custom \*\*LLM-as-judge evaluation framework\*\*

(faithfulness, answer relevance, context precision).



\## Key features



\### Role-Based Access Control (RBAC)

Access control is enforced \*\*inside the vector database query itself\*\*, not

as a prompt instruction. Each document chunk is tagged with a `department`

field at ingestion time; each user role maps to a list of allowed

departments; Qdrant's metadata filter ensures out-of-scope chunks are never

even retrieved, let alone shown to the LLM.



```python

rbac\_filter = Filter(

&#x20;   must=\[FieldCondition(key="department", match=MatchAny(any=allowed\_departments))]

)

```



This means a chunk the user isn't authorized for cannot leak through — even

via a cleverly-phrased question — because the LLM never receives it.



\### Guardrails

\- \*\*Out-of-scope detection\*\* — if the top retrieval similarity score falls

&#x20; below a threshold, the question is refused \*before\* an LLM call is made,

&#x20; saving cost and preventing off-topic answers.

\- \*\*PII detection\*\* — every retrieved context is scanned with Microsoft

&#x20; Presidio, flagging entity types (names, emails, dates, etc.) present in the

&#x20; source material shown to the user.



\### Evaluation

A custom evaluation harness (avoiding a broken upstream dependency in Ragas)

implements the same core LLM-as-judge pattern:

\- \*\*Faithfulness\*\* — does the generated answer stick to what the retrieved

&#x20; context actually supports?

\- \*\*Answer relevance\*\* — does the answer address the question asked?

\- \*\*Context precision\*\* — were the retrieved chunks actually useful?



\### Observability

Every step of the pipeline (retrieval, prompt construction, LLM call) is

wrapped in LangSmith `@traceable` decorators, giving a full waterfall view of

latency, token usage, and intermediate data for any given request.



\## A real bug found (and fixed) via evaluation



Early evaluation runs caught a genuine RAG failure mode: asking \*"How many

employees are there?"\* returned \*\*"Only 1 employee"\*\* instead of the correct

answer (100). Faithfulness scored 1.0 (the LLM didn't hallucinate — it

genuinely couldn't see the full dataset), but Context Precision scored 0.0,

correctly identifying the retrieval was the problem, not the generation.



\*\*Root cause:\*\* vanilla RAG retrieves only the top-k similar chunks. A

100-row CSV split into many chunks has no single chunk that "knows" the

total row count — retrieval could only ever surface a handful of rows.



\*\*Fix:\*\* pre-compute aggregate statistics (counts, averages, breakdowns) from

the raw tabular data and inject them as their own retrievable summary

document. After the fix, the same question correctly answered "100 total

employees," with all three evaluation metrics at or near 1.0.



This is a common, well-documented limitation of naive RAG over tabular data

— production systems typically solve it with either this summary-injection

pattern or a query router that sends aggregate-style questions to SQL/pandas

instead of vector search.



\## Tech stack



| Layer | Technology |

|---|---|

| Document parsing | Docling, pandas |

| Chunking | LangChain (RecursiveCharacterTextSplitter) |

| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local, free) |

| Vector store | Qdrant (local mode) |

| LLM | Groq (Llama 3.3 70B, free tier) |

| PII detection | Microsoft Presidio |

| Observability | LangSmith |

| Evaluation | Custom LLM-as-judge harness |

| Frontend | Streamlit |



\## Project structure



atliq-rag/

├── data/

│ ├── hr/ # HR documents + auto-generated summary stats

│ ├── finance/

│ ├── marketing/

│ ├── engineering/

│ └── general/

├── src/

│ ├── 01\_ingest\_documents.py # loads .md/.csv/.xlsx, tags by department

│ ├── 02\_chunk\_documents.py # splits documents into retrievable chunks

│ ├── 03\_embed\_and\_store.py # embeds chunks, builds Qdrant collection

│ ├── 04\_ask\_question.py # baseline RAG (no RBAC)

│ ├── 05\_rbac\_chat.py # adds role-based retrieval filtering

│ ├── 06\_guardrails.py # adds out-of-scope + PII detection

│ ├── 07\_evaluate\_and\_trace.py # LangSmith tracing

│ ├── 08\_custom\_eval.py # LLM-as-judge evaluation (faithfulness, etc.)

│ ├── 09\_generate\_summaries.py # aggregate-stats generator (bug fix)

│ └── 10\_app.py # Streamlit chat UI

└── qdrant\_storage/ # local vector DB (gitignored, regenerable)





\## Running it locally



```bash

python -m venv .venv

.venv\\Scripts\\Activate.ps1        # Windows

pip install -r requirements.txt   # or install packages per script as needed



\# Add your Groq API key to .env:

\# GROQ\_API\_KEY=your\_key\_here



python src/03\_embed\_and\_store.py  # builds the vector store

streamlit run src/10\_app.py       # launches the chat UI

```

