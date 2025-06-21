#!/usr/bin/env python3
# faiss-ingest.py - Load PDFs and infer classification from folder structure for embedding and indexing
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install langchain-community oci PyPDF2
#   - Ensure OCI CLI config is set up in ~/.oci/config
#   - Recommended: run ./classify-docs.py to classify the PDFs before ingesting.
#
# Usage:
#   python faiss-ingest.py                       # Load folders from ksources.txt
#   python faiss-ingest.py --input ./folder1     # Specify one or more folders or files
#   python faiss-ingest.py --input ./doc.pdf     # Specify a single PDF
#   python faiss-ingest.py --debug               # Enable verbose logging
#   python faiss-ingest.py --input ./dir --debug # Combine input and debug options
#   python faiss-ingest.py --input /path/to/pdf_folder [--debug]

import os
import sys
import contextlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import importlib.util, os

# dynamic load of configuration loader (load-config.py)
spec_cfg = importlib.util.spec_from_file_location(
    "load_config", os.path.join(os.path.dirname(__file__), "load-config.py")
)
cfg_mod = importlib.util.module_from_spec(spec_cfg)
spec_cfg.loader.exec_module(cfg_mod)
LoadConfig = cfg_mod.LoadConfig
import argparse

def load_sources(file_path="ksources.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' not found. Please create this file with one directory path per line.")
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

properties = LoadConfig()

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--input", type=str, nargs="*", help="Paths to folders or files")
args = parser.parse_args()
DEBUG = args.debug

input_paths = args.input if args.input else load_sources()

pdf_file_count = 0

def normalise(text):
    return text.strip().lower().replace(" ", "_").replace("-", "_")

def infer_from_folder(file_path):
    path = Path(file_path).resolve()
    for parent in path.parents:
        parts = parent.name.split("-")
        if len(parts) == 2:
            return normalise(parts[0]), normalise(parts[1])
    return "unclassified", "unclassified"

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
                            for doc in docs:
                                doc.metadata.update({
                                    "source": full_path,
                                    "audience": audience,
                                    "type": doc_type
                                })
                            documents.extend(docs)
                            pdf_file_count += 1
                            if DEBUG:
                                print(f"Ingested: {full_path} ({audience}, {doc_type})")
                        except Exception as e:
                            if DEBUG:
                                print(f"[WARN] Failed to load {full_path}: {e}")
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
                for doc in docs:
                    doc.metadata.update({
                        "source": path,
                        "audience": audience,
                        "type": doc_type
                    })
                documents.extend(docs)
                pdf_file_count += 1
                if DEBUG:
                    print(f"Ingested: {path} ({audience}, {doc_type})")
            except Exception as e:
                if DEBUG:
                    print(f"[WARN] Failed to load {path}: {e}")
    return documents

# --- Entry point ---
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
