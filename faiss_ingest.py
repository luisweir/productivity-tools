#!/usr/bin/env python3
# faiss_ingest.py - Load PDFs and infer classification from folder structure for embedding and indexing
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install langchain-community oci PyPDF2
#   - Ensure OCI CLI config is set up in ~/.oci/config
#   - Recommended: run ./classify_docs.py to classify the PDFs before ingesting.
#
# Usage:
#   python faiss_ingest.py                       # Load folders from ksources.txt
#   python faiss_ingest.py --input ./folder1     # Specify one or more folders or files
#   python faiss_ingest.py --input ./doc.pdf     # Specify a single PDF
#   python faiss_ingest.py --debug               # Enable verbose logging
#   python faiss_ingest.py --input ./dir --debug # Combine input and debug options
#   python faiss_ingest.py --input /path/to/pdf_folder [--debug]

import os
import contextlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import argparse
from load_config import LoadConfig

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--input", type=str, nargs="*", help="Paths to folders or files")
args = parser.parse_args()
DEBUG = args.debug

# Load properties
properties = LoadConfig()

# Load source paths
def load_sources(file_path="ksources.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' not found. Please create this file with one directory path per line.")
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

input_paths = args.input if args.input else load_sources()
pdf_file_count = 0

# Utility functions
def normalise(text):
    return text.strip().lower().replace(" ", "_").replace("-", "_")

def infer_from_folder(file_path):
    path = Path(file_path).resolve()
    for parent in path.parents:
        parts = parent.name.split("-")
        if len(parts) == 2:
            return normalise(parts[0]), normalise(parts[1])
    return "unclassified", "unclassified"

# classify documents as "Oracle Owned" when they're Oracle's
def is_oracle_owned(path: str, DEBUG: bool = False) -> bool:
    from PyPDF2 import PdfReader
    import os
    from pathlib import Path

    content_indicators = [
        "oracle confidential",
        "internal use only",
        "© oracle",
        "oracle and/or its affiliates",
        "oracle corporation",
        "copyright © oracle",
        "oracle cloud infrastructure",
        "oracle fusion",
        "oracle hospitality"
    ]
    file_indicators = [
        "oci", "ohip", "fusion", "myoracle", "opera", "micros",
        "orcl", "ocs", "cloud_console", "cloud_ops", "internal_tool"
    ]

    try:
        # Check the last folder in the path
        last_folder = Path(path).parent.name.lower()
        if DEBUG:
            print(f"[DEBUG] Last folder: {last_folder}")
        if "internal" in last_folder or "oracle" in last_folder:
            if DEBUG:
                print("[DEBUG] Match found in last folder name")
            return True

        # Check file name
        filename = os.path.basename(path).lower()
        if DEBUG:
            print(f"[DEBUG] Checking file name: {filename}")
        if any(kw in filename for kw in file_indicators):
            if DEBUG:
                print(f"[DEBUG] Match found in file name")
            return True

        # Check PDF content (first 5 pages)
        reader = PdfReader(path)
        text_found = ""
        for i, page in enumerate(reader.pages[:5]):
            text = page.extract_text() or ""
            if DEBUG:
                print(f"[DEBUG] Page {i+1} text: {repr(text[:200])}")
            text_found += text.lower()

        if any(ind in text_found for ind in content_indicators):
            if DEBUG:
                print("[DEBUG] Match found in PDF content")
            return True

    except Exception as e:
        if DEBUG:
            print(f"[WARN] Failed to process {path}: {e}")

    if DEBUG:
        print(f"[DEBUG] No Oracle indicators found in file: {path}")
    return False

# Load documents
def load_documents_from_folders(paths):
    global pdf_file_count
    documents = []
    for path in paths:
        if os.path.isdir(path):
            print(f"Scanning directory: {path}")
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        full_path = os.path.join(root, file)
                        if DEBUG:
                            print(f"Found: {full_path}")
                        audience, doc_type = infer_from_folder(full_path)
                        try:
                            with open(os.devnull, "w") as devnull:
                                with contextlib.redirect_stderr(devnull):
                                    loader = PyPDFLoader(full_path)
                                    docs = loader.load()
                            oracle_flag = bool(is_oracle_owned(full_path, DEBUG))
                            for doc in docs:
                                doc.metadata.update({
                                    "source": full_path or path,
                                    "audience": audience,
                                    "type": doc_type,
                                    "oracle_owned": oracle_flag,
                                    "chunk_count": len(docs)
                                })
                                if DEBUG:
                                    print(f"[DEBUG] Metadata added -> source: {doc.metadata.get('source')}, "
                                        f"audience: {doc.metadata.get('audience')}, type: {doc.metadata.get('type')}, "
                                        f"oracle_owned: {doc.metadata.get('oracle_owned')} ({type(doc.metadata.get('oracle_owned'))})")
                        
                            documents.extend(docs)
                            pdf_file_count += 1
                            if DEBUG:
                                print(f"Ingested: {path} ({audience}, {doc_type}, {oracle_flag})")
                        except Exception as e:
                            if DEBUG:
                                print(f"[WARN] Failed to load {path}: {e}")
        elif os.path.isfile(path) and path.lower().endswith(".pdf"):
            print(f"Scanning file: {path}")
            if DEBUG:
                print(f"Found: {path}")
            audience, doc_type = infer_from_folder(path)
            try:
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stderr(devnull):
                        loader = PyPDFLoader(path)
                        docs = loader.load()
                oracle_flag = bool(is_oracle_owned(full_path, DEBUG))
                for doc in docs:
                    doc.metadata.update({
                        "source": full_path or path,
                        "audience": audience,
                        "type": doc_type,
                        "oracle_owned": oracle_flag,
                        "chunk_count": len(docs)
                    })
                    if DEBUG:
                        print(f"[DEBUG] Metadata added -> source: {doc.metadata.get('source')}, "
                            f"audience: {doc.metadata.get('audience')}, type: {doc.metadata.get('type')}, "
                            f"oracle_owned: {doc.metadata.get('oracle_owned')} ({type(doc.metadata.get('oracle_owned'))})")
    
                documents.extend(docs)
                pdf_file_count += 1
                if DEBUG:
                    print(f"Ingested: {path} ({audience}, {doc_type}, {oracle_flag})")
            except Exception as e:
                if DEBUG:
                    print(f"[WARN] Failed to load {path}: {e}")
    return documents

# Entry point
if __name__ == "__main__":
    documents = load_documents_from_folders(input_paths)

    print(f"\nPDF files successfully ingested: {pdf_file_count}")
    print(f"Loaded document objects: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=300)
    all_documents = text_splitter.split_documents(documents)
    print(f"Total chunks: {len(all_documents)}")

    if not all_documents:
        print("No documents to process. Exiting.")
        exit()

    embeddings = OCIGenAIEmbeddings(
        model_id=properties.getEmbeddingModelName(),
        service_endpoint=properties.getEndpoint(),
        compartment_id=properties.getCompartment(),
        model_kwargs={"truncate": True}
    )

    batch_size = 96
    db = FAISS.from_documents(all_documents[:batch_size], embeddings)
    print(f"Indexed initial {min(batch_size, len(all_documents))} documents")

    for i in range(1, (len(all_documents) + batch_size - 1) // batch_size):
        start = i * batch_size
        end = start + batch_size
        db.add_documents(all_documents[start:end])
        print(f"Indexed documents {start} to {end}")

    db.save_local("faiss_index")
    print("FAISS index saved successfully.")
