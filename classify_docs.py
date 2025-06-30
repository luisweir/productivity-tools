#!/usr/bin/env python3
# classify_docs.py - Recursively scan PDFs, classify via OCI Generative AI, and organise into subfolders by category.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install langchain-community oci PyPDF2
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Usage:
#   python classify_docs.py --input /path/to/pdf_folder                    # classify all PDFs in a folder
#   python classify_docs.py --input file1.pdf file2.pdf /path/to/folder     # classify multiple PDFs or folders
#   python classify_docs.py --debug --input /path/to/pdf_folder             # enable debug logging

#!/usr/bin/env python3
# classify_docs.py - Recursively scan PDFs, classify via OCI Generative AI, and organise into subfolders by category.

import os
import shutil
import re
import warnings
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from load_config import LoadConfig
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails, TextContent, Message, GenericChatRequest,
    OnDemandServingMode, BaseChatRequest
)
from oci.retry import NoneRetryStrategy
import argparse
import glob

warnings.filterwarnings("ignore", category=UserWarning)

properties = LoadConfig()

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--input", type=str, nargs='+', help="Path(s) to PDF files or folders")
args = parser.parse_args()
DEBUG = args.debug

from chat_engine import ChatEngine

ALLOWED_AUDIENCES = ChatEngine.ALLOWED_AUDIENCES
ALLOWED_TYPES = ChatEngine.ALLOWED_TYPES

def normalise(text):
    return text.strip().lower().replace(" ", "_").replace("-", "_")

def is_classified_folder(folder_name):
    audiences = "|".join(re.escape(a) for a in ALLOWED_AUDIENCES.keys())
    types = "|".join(re.escape(t) for t in ALLOWED_TYPES.keys())
    pattern = rf"^({audiences})-({types})$"
    return bool(re.match(pattern, folder_name, re.IGNORECASE))

def load_sources_from_file(file_path="ksources.txt"):
    if not os.path.isfile(file_path):
        print(f"[ERROR] No input provided and '{file_path}' not found.")
        return []
    sources = []
    print(f"[DEBUG] Reading sources from {file_path}...")
    with open(file_path, "r") as f:
        for line in f:
            raw_line = line.rstrip()
            print(f"[DEBUG] Raw line: '{raw_line}'")
            line = raw_line.strip()
            if not line or line.startswith("#"):
                print(f"[DEBUG] Skipping line: '{raw_line}'")
                continue
            expanded = glob.glob(line)
            print(f"[DEBUG] Expanded path(s) from line: {expanded}")
            sources.extend(expanded)
    print(f"[DEBUG] Total valid sources loaded: {sources}")
    return sources

def get_oci_client():
    config = oci.config.from_file("~/.oci/config", properties.getDefaultProfile())
    return GenerativeAiInferenceClient(
        config=config,
        service_endpoint=properties.getEndpoint(),
        retry_strategy=NoneRetryStrategy(),
        timeout=(10, 240)
    )

def extract_sample_text(file_path, max_chars=1000):
    if DEBUG:
        print(f"[DEBUG] Attempting to load PDF: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        print(f"[ERROR] Failed to load PDF '{file_path}': {e}")
        return "Document could not be loaded.", 0

    if DEBUG:
        print(f"[DEBUG] Loaded {len(docs)} pages from {file_path}")

    title = Path(file_path).stem.replace("_", " ").replace("-", " ")
    all_text = " ".join([doc.page_content.strip() for doc in docs])
    word_count = len(all_text.split())

    sample_text = "\n".join(doc.page_content.strip() for doc in docs[:3])
    if len(sample_text) > max_chars:
        sample_text = sample_text[:max_chars]

    result = f"Document Title: {title}\nSample Content (Pages 1–3):\n{sample_text}"
    return result, word_count

def get_existing_classifications(base_dir):
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.count("-") == 1]
    if DEBUG:
        print(f"[DEBUG] Existing classification folders in {base_dir}: {folders}")
    return folders

def find_closest_match(audience, doc_type, existing_classes):
    target = f"{audience}-{doc_type}"
    norm_target = normalise(target)
    for existing in existing_classes:
        if normalise(existing) == norm_target:
            return existing
    return None

def classify_with_genai(text_sample, existing_classes, word_count):
    audience_expl = "\n".join([f"- {k}: {v}" for k, v in ALLOWED_AUDIENCES.items()])
    type_expl = "\n".join([f"- {k}: {v}" for k, v in ALLOWED_TYPES.items()])
    normalised_classes = [normalise(c) for c in existing_classes]

    prompt = (
        "You are an AI classification assistant. Your task is to classify the document below into two categories:\n"
        "1. Audience\n"
        "2. Type\n\n"
        f"Valid Audience values:\n{audience_expl}\n\n"
        f"Valid Type values:\n{type_expl}\n\n"
        f"The document is approximately {word_count} words long.\n"
        "If the document is longer than 2,500 words, it must be either 'deepdive' or 'research':\n"
        "If the document clearly identifies itself as a research paper, classify it as 'research'.\n"
        "- Use 'deepdive' if it includes practical implementation guidance, system architecture, or detailed processes.\n"
        "- Use 'research' if it focuses on analysis, experiments, findings, hypotheses, or has an academic style.\n\n"
        "Return only the classification using this format:\n"
        "`audience_type`\n"
        "Do NOT explain, comment, or use any words beyond the label.\n\n"
        f"If there are existing classification folders, they are: {', '.join(normalised_classes)}\n"
        "Use an existing one if it matches. Otherwise, return a new one if clearly required.\n\n"
        f"Document content:\n{text_sample}\n\n"
        "Your response:"
    )

    if DEBUG:
        print("\n[DEBUG] Classification Prompt:\n" + prompt)

    client = get_oci_client()
    content = TextContent(text=prompt)
    message = Message(role="USER", content=[content])
    chat_request = GenericChatRequest(
        api_format=BaseChatRequest.API_FORMAT_GENERIC,
        messages=[message],
        max_tokens=200,
        temperature=0.2,
        frequency_penalty=0,
        presence_penalty=0,
        top_p=1,
        top_k=0
    )
    chat_detail = ChatDetails(
        serving_mode=OnDemandServingMode(model_id=properties.getModelOcid()),
        chat_request=chat_request,
        compartment_id=properties.getCompartment()
    )
    response = client.chat(chat_detail)
    raw = response.data.chat_response.choices[0].message.content[0].text.strip()

    if DEBUG:
        print("[DEBUG] Gen AI Response:\n" + raw + "\n")

    raw_clean = raw.strip().lower().replace("-", "_").replace(" ", "_")
    audiences = "|".join(re.escape(k) for k in ALLOWED_AUDIENCES.keys())
    types = "|".join(re.escape(k) for k in ALLOWED_TYPES.keys())
    pattern = rf"^({audiences})_({types})$"
    match = re.match(pattern, raw_clean)

    if match:
        return match.group(1), match.group(2)

    print(f"[WARN] Could not parse valid classification from response: '{raw}'")
    return "unclassified", "unclassified"

def reclassify_and_move(file_path, root_input):
    if DEBUG:
        print(f"[DEBUG] Processing file: {file_path}")

    sample, word_count = extract_sample_text(file_path)
    existing_classes = get_existing_classifications(root_input)
    audience, doc_type = classify_with_genai(sample, existing_classes, word_count)
    matched_folder = find_closest_match(audience, doc_type, existing_classes)
    target_folder_name = matched_folder or f"{audience}-{doc_type}"
    new_folder = os.path.join(root_input, target_folder_name)
    os.makedirs(new_folder, exist_ok=True)
    new_path = os.path.join(new_folder, os.path.basename(file_path))
    shutil.move(file_path, new_path)
    if DEBUG:
        print(f"[DEBUG] Moved '{os.path.basename(file_path)}' to '{new_folder}'")
    return new_path, audience, doc_type

def classify_files(paths):
    if DEBUG:
        print(f"[DEBUG] Input paths to process: {paths}")
    for root_input in paths:
        if os.path.isdir(root_input):
            print(f"Scanning directory: {root_input}")
            for root, dirs, files in os.walk(root_input):
                if DEBUG:
                    print(f"[DEBUG] Entering directory: {root}")
                    print(f"[DEBUG] Files found: {files}")
                folder_name = os.path.basename(root)
                if is_classified_folder(folder_name):
                    if DEBUG:
                        print(f"[DEBUG] Skipping classified folder: {folder_name}")
                    continue
                for file in files:
                    if file.lower().endswith(".pdf"):
                        file_path = os.path.join(root, file)
                        reclassify_and_move(file_path, root_input)
        elif os.path.isfile(root_input) and root_input.lower().endswith(".pdf"):
            parent_folder = os.path.basename(os.path.dirname(root_input))
            if is_classified_folder(parent_folder):
                if DEBUG:
                    print(f"[DEBUG] Skipping file already in classified folder: {root_input}")
            else:
                reclassify_and_move(root_input, os.path.dirname(root_input))

# Entry Point
if __name__ == "__main__":
    input_paths = args.input if args.input else load_sources_from_file()
    if not input_paths:
        print("No input paths provided or found. Exiting.")
    else:
        classify_files(input_paths)
