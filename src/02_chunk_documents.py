"""
Step 2: Chunking - split each loaded document into smaller pieces
so retrieval can find precisely relevant sections instead of whole documents.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from importlib import import_module
import sys
sys.path.append("src")
ingest = import_module("01_ingest_documents")


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for doc in documents:
        text_pieces = splitter.split_text(doc["text"])

        for i, piece in enumerate(text_pieces):
            all_chunks.append({
                "text": piece,
                "department": doc["department"],
                "filename": doc["filename"],
                "source_path": doc["source_path"],
                "chunk_index": i,
            })

    return all_chunks


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Loading documents...")
    print("=" * 60)
    documents = ingest.load_all_documents()
    print(f"Loaded {len(documents)} documents")

    print()
    print("=" * 60)
    print("Step 2: Chunking documents...")
    print("=" * 60)
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    print()
    print("Sample chunks:")
    print("-" * 60)
    for chunk in chunks[:3]:
        print(f"[{chunk['department']}] {chunk['filename']} (chunk #{chunk['chunk_index']})")
        print(f"  Length: {len(chunk['text'])} chars")
        print(f"  Text: {chunk['text'][:150]}...")
        print()

    print("-" * 60)
    print("Chunk count per department:")
    dept_counts = {}
    for chunk in chunks:
        dept_counts[chunk["department"]] = dept_counts.get(chunk["department"], 0) + 1
    for dept, count in dept_counts.items():
        print(f"  {dept:12} -> {count} chunks")