# faiss-ingest.py - Load and split PDFs into chunks, then index into a FAISS vector store using Oracle embeddings.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install langchain langchain-community PyPDF2 faiss-cpu oci
#   - Ensure config.txt exists in the same directory with OCI and LangChain settings
#
# Usage:
#   - Create a 'sources.txt' file in the same directory, with one line per PDF folder path to be indexed
#   - Run the script:
#       python faiss-ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from LoadProperties import LoadProperties

def normalise(text):
    return text.strip().lower().replace(" ", "_")

def load_sources(file_path="ksources.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' not found. Please create this file with one directory path per line.")
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def load_pdfs_recursively(base_paths):
    all_documents = []
    for base_path in base_paths:
        print(f"\nScanning directory: {base_path}")
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    try:
                        print(f"Loading: {file_path}")
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source"] = file_path
                        all_documents.extend(docs)
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")
    return all_documents

def infer_context_from_path(file_path: str) -> dict:
    folder_name = os.path.basename(os.path.dirname(file_path))
    parts = folder_name.split("--")
    if len(parts) != 3:
        return {"audience": "unknown", "domain": "unknown", "type": "unknown"}

    audience = parts[0].split("_", 1)[-1]
    domain = parts[1]
    doc_type = parts[2]

    return {
        "audience": normalise(audience),
        "domain": normalise(domain),
        "type": normalise(doc_type)
    }

# Load directories from sources.txt
pdf_dirs = load_sources()

# Load PDFs
documents = load_pdfs_recursively(pdf_dirs)
print(f"\nLoaded PDF count: {len(documents)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

# Enrich metadata per chunk
for doc in all_documents:
    file_path = doc.metadata.get("source", "unknown")
    inferred = infer_context_from_path(file_path)
    doc.metadata["audience"] = inferred["audience"]
    doc.metadata["domain"] = inferred["domain"]
    doc.metadata["type"] = inferred["type"]

print(f"Total chunks: {len(all_documents)}")

# Abort if no content
if not all_documents:
    print("No documents to process. Exiting.")
    exit()

# Load embedding model
properties = LoadProperties()
embeddings = OCIGenAIEmbeddings(
    model_id=properties.getEmbeddingModelName(),
    service_endpoint=properties.getEndpoint(),
    compartment_id=properties.getCompartment(),
    model_kwargs={"truncate": True}
)

# Create FAISS index
batch_size = 96
db = FAISS.from_documents(all_documents[:batch_size], embeddings)
print(f"Indexed initial {min(batch_size, len(all_documents))} documents")

for i in range(1, (len(all_documents) + batch_size - 1) // batch_size):
    start = i * batch_size
    end = start + batch_size
    batch = all_documents[start:end]
    db.add_documents(batch)
    print(f"Indexed documents {start} to {end}")

# Save FAISS index
db.save_local("faiss_index")
print("FAISS index saved successfully.")
