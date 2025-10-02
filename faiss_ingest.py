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
from langchain.schema import Document
import argparse
from load_config import LoadConfig
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails, TextContent, Message, GenericChatRequest,
    OnDemandServingMode, BaseChatRequest
)
from oci.retry import NoneRetryStrategy

from rag_toolkit import ChatEngine

# Bring in the allowed audiences/types from rag_toolkit so classification prompts are consistent
ALLOWED_AUDIENCES = ChatEngine.ALLOWED_AUDIENCES
ALLOWED_TYPES = ChatEngine.ALLOWED_TYPES

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

def get_oci_client():
    props = LoadConfig()
    config = oci.config.from_file("~/.oci/config", props.getDefaultProfile())
    return GenerativeAiInferenceClient(
        config=config,
        service_endpoint=props.getEndpoint(),
        retry_strategy=NoneRetryStrategy(),
        timeout=(10, 240),
    )


def is_oracle_owned(path: str, DEBUG: bool = False) -> bool:
    """
    Use an OCI Generative AI model to classify whether the document is Oracle-owned.
    The function extracts a sample from the document (PDF or TXT) and asks the model
    to reply with a single token: 'oracle' or 'external'.
    """
    import os
    from pathlib import Path
    from PyPDF2 import PdfReader

    sample_text = ""
    try:
        if str(path).lower().endswith(".pdf"):
            reader = PdfReader(path)
            pages = reader.pages[:5]
            sample_text = "\n".join((p.extract_text() or "") for p in pages)
        elif str(path).lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                sample_text = f.read(4000)
        else:
            # fallback: try to read as text
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                sample_text = f.read(4000)
    except Exception as e:
        if DEBUG:
            print(f"[WARN] Failed to extract sample from {path}: {e}")
        return False

    # Build prompt
    prompt = (
        "You are an assistant that must determine whether a document was authored or published by Oracle.\n"
        "Given the document sample below, reply with exactly one word: 'oracle' if the content is Oracle-owned/authored, "
        "or 'external' if it is not. If you are unsure, answer 'external'.\n\n"
        f"Document sample:\n{sample_text}\n\nYour answer:"
    )

    if DEBUG:
        print("[DEBUG] Oracle ownership prompt length:", len(prompt))

    try:
        client = get_oci_client()
        content = TextContent(text=prompt)
        message = Message(role="USER", content=[content])
        chat_request = GenericChatRequest(
            api_format=BaseChatRequest.API_FORMAT_GENERIC,
            messages=[message],
        )
        props = LoadConfig()
        chat_detail = ChatDetails(
            serving_mode=OnDemandServingMode(model_id=props.getModelOcid()),
            chat_request=chat_request,
            compartment_id=props.getCompartment(),
        )
        response = client.chat(chat_detail)
        raw = response.data.chat_response.choices[0].message.content[0].text.strip().lower()
        if DEBUG:
            print("[DEBUG] Oracle ownership classifier response:\n", raw)
        if raw.startswith("oracle"):
            return True
        return False
    except Exception as e:
        if DEBUG:
            print(f"[WARN] Oracle ownership classifier failed: {e}")
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
                    full_path = os.path.join(root, file)
                    if file.lower().endswith(".pdf"):
                        if DEBUG:
                            print(f"Found PDF: {full_path}")
                        audience, doc_type = infer_from_folder(full_path)
                        try:
                            with open(os.devnull, "w") as devnull:
                                with contextlib.redirect_stderr(devnull):
                                    loader = PyPDFLoader(full_path)
                                    docs = loader.load()
                            oracle_flag = bool(is_oracle_owned(full_path, DEBUG))
                            for doc in docs:
                                doc.metadata.update({
                                    "source": full_path,
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
                                print(f"Ingested: {full_path} ({audience}, {doc_type}, {oracle_flag})")
                        except Exception as e:
                            if DEBUG:
                                print(f"[WARN] Failed to load {full_path}: {e}")
                    elif file.lower().endswith(".txt"):
                        # simple text ingestion
                        if DEBUG:
                            print(f"Found TXT: {full_path}")
                        audience, doc_type = infer_from_folder(full_path)
                        try:
                            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                            oracle_flag = bool(is_oracle_owned(full_path, DEBUG))
                            doc = Document(page_content=content, metadata={
                                "source": full_path,
                                "audience": audience,
                                "type": doc_type,
                                "oracle_owned": oracle_flag,
                                "chunk_count": 1,
                            })
                            documents.append(doc)
                            pdf_file_count += 1
                            if DEBUG:
                                print(f"Ingested TXT: {full_path} ({audience}, {doc_type}, {oracle_flag})")
                        except Exception as e:
                            if DEBUG:
                                print(f"[WARN] Failed to load TXT {full_path}: {e}")
        elif os.path.isfile(path) and path.lower().endswith((".pdf", ".txt")):
            print(f"Scanning file: {path}")
            if DEBUG:
                print(f"Found: {path}")
            audience, doc_type = infer_from_folder(path)
            try:
                if path.lower().endswith(".pdf"):
                    with open(os.devnull, "w") as devnull:
                        with contextlib.redirect_stderr(devnull):
                            loader = PyPDFLoader(path)
                            docs = loader.load()
                    oracle_flag = bool(is_oracle_owned(path, DEBUG))
                    for doc in docs:
                        doc.metadata.update({
                            "source": path,
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
                else:
                    # txt file
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    oracle_flag = bool(is_oracle_owned(path, DEBUG))
                    doc = Document(page_content=content, metadata={
                        "source": path,
                        "audience": audience,
                        "type": doc_type,
                        "oracle_owned": oracle_flag,
                        "chunk_count": 1,
                    })
                    documents.append(doc)
                    pdf_file_count += 1
                    if DEBUG:
                        print(f"Ingested TXT: {path} ({audience}, {doc_type}, {oracle_flag})")
            except Exception as e:
                if DEBUG:
                    print(f"[WARN] Failed to load {path}: {e}")
    return documents

# Entry point
if __name__ == "__main__":
    documents = load_documents_from_folders(input_paths)

    print(f"\nFiles successfully ingested: {pdf_file_count}")
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
