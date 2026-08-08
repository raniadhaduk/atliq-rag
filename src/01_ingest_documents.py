"""
Step 1: Ingestion - load every document from data/<department>/ folders
and tag each one with its department, based on the folder it lives in.

Supports .md, .xlsx, and .csv files.

Run with:  python src\01_ingest_documents.py
"""

import os
from pathlib import Path
import pandas as pd
from docling.document_converter import DocumentConverter

DATA_DIR = Path("data")
DEPARTMENTS = ["marketing", "engineering", "finance", "general", "hr"]
SUPPORTED_EXTENSIONS = {".md", ".xlsx", ".csv"}


def load_all_documents():
    """
    Walk every department folder, read each supported file,
    and return a list of dicts like:
        {
            "text": "...",
            "department": "hr",
            "filename": "hr_policy.md",
            "source_path": "data/hr/hr_policy.md"
        }
    """
    converter = DocumentConverter()  # Docling's main entry point (used for .xlsx)
    documents = []

    for dept in DEPARTMENTS:
        dept_folder = DATA_DIR / dept

        if not dept_folder.exists():
            print(f"  [skip] No folder found for department: {dept}")
            continue

        for file_path in dept_folder.iterdir():
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue  # skip file types we don't handle yet

            print(f"  Loading: {file_path}")

            suffix = file_path.suffix.lower()

            if suffix == ".md":
                # Markdown is already plain text - just read it directly
                text = file_path.read_text(encoding="utf-8")

            elif suffix == ".xlsx":
                # Docling converts Excel into clean markdown, preserving table structure
                result = converter.convert(str(file_path))
                text = result.document.export_to_markdown()

            elif suffix == ".csv":
                # Read CSV into a DataFrame, then render as a markdown table
                # so it's easy for the LLM to read later
                df = pd.read_csv(file_path)
                text = df.to_markdown(index=False)

            else:
                continue

            documents.append({
                "text": text,
                "department": dept,          # <-- THE metadata field for RBAC
                "filename": file_path.name,
                "source_path": str(file_path),
            })

    return documents


if __name__ == "__main__":
    print("=" * 60)
    print("Ingesting documents from data/<department>/ folders...")
    print("=" * 60)

    docs = load_all_documents()

    print()
    print(f"Total documents loaded: {len(docs)}")
    print()

    # Quick sanity check: show department + filename + first 100 chars of each
    for doc in docs:
        preview = doc["text"][:100].replace("\n", " ")
        print(f"[{doc['department']:10}] {doc['filename']:25} -> {preview}...")