#!/usr/bin/env python3
# organise_pdfs.py - Recursively scan PDFs, classify via OCI Generative AI, and organize into subfolders by category.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install PyPDF2 oci
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Usage:
#   python organise_pdfs.py -p /path/to/pdf_folder

import sys
import argparse
import shutil
from pathlib import Path

import PyPDF2
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    TextContent, Message, GenericChatRequest, ChatDetails,
    OnDemandServingMode, BaseChatRequest
)
from oci.retry import NoneRetryStrategy
from LoadProperties import LoadProperties

# Load properties
properties = LoadProperties()

def get_oci_client():
    config = oci.config.from_file('~/.oci/config', properties.getDefaultProfile())
    return GenerativeAiInferenceClient(
        config=config,
        service_endpoint=properties.getEndpoint(),  # If endpoint is a property, no parentheses
        retry_strategy=NoneRetryStrategy(),
        timeout=(10, 240)
    )


def extract_text_sample(pdf_path: Path, max_pages: int = 2) -> str:
    """
    Extract the first max_pages pages of text from a PDF as a single string.
    """
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            try:
                text.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(text).strip()


def chat_with_genai(prompt: str, max_tokens: int = 300) -> str:
    client = get_oci_client()
    """
    Send a single-user prompt to OCI Generative AI and return the text response.
    """
    content = TextContent(text=prompt)
    message = Message(role="USER", content=[content])
    chat_req = GenericChatRequest(
        api_format=BaseChatRequest.API_FORMAT_GENERIC,
        messages=[message],
        max_tokens=max_tokens,
        temperature=0.3,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        top_p=1.0
    )
    details = ChatDetails(
        serving_mode=OnDemandServingMode(model_id=properties.getModelOcid()),
        chat_request=chat_req,
        compartment_id=properties.getCompartment()
    )
    resp = client.chat(details)
    return resp.data.chat_response.choices[0].message.content[0].text.strip()


def infer_categories(sample_texts: list[str], max_cats: int = 5) -> list[str]:
    """
    Ask GenAI to generate categories based on audience, domain, and document type.
    """
    examples = "\n\n".join(
        f"{i+1}) “{txt[:300].replace(chr(10), ' ')}…”"
        for i, txt in enumerate(sample_texts)
    )
    prompt = f"""
Given these excerpts from PDF documents related to AI and Machine Learning, 
categorise them across three dimensions:

1. Target audience (e.g., Executives, Engineers, Product Managers, Industry Experts)
2. Domain/subdomain (e.g., NLP, Computer Vision, AI Governance, Time Series Forecasting)
3. Document type (e.g., Research Paper, Whitepaper, Product Documentation)

Suggest up to {max_cats} meaningful combinations in the format:

<Audience> | <Domain> | <Type>

Return only the list of categorisation labels, one per line.
    
Here are the samples:

{examples}
"""
    ai_text = chat_with_genai(prompt, max_tokens=400)
    cats = [line.strip() for line in ai_text.splitlines() if line.strip()]
    return list(dict.fromkeys(cats))


def classify_document(doc_text: str, categories: list[str]) -> str:
    """
    Classify the document into one of the inferred categories.
    """
    cat_list = "\n".join(f"{i+1}) {c}" for i, c in enumerate(categories))
    prompt = (
        f"From the following categories, select the one that best fits this document. "
        f"Each category includes audience, domain, and type.\n\n"
        f"Categories:\n{cat_list}\n\n"
        f"Document excerpt:\n{doc_text[:500].replace(chr(10), ' ')}\n\n"
        f"Return only the exact category label."
    )
    ai_text = chat_with_genai(prompt, max_tokens=80)
    for c in categories:
        if c.lower() in ai_text.lower():
            return c
    return categories[0]


def main(root_folder: str):
    root = Path(root_folder)
    if not root.is_dir():
        print(f"Error: {root} is not a folder.")
        sys.exit(1)

    # 1) collect all PDFs
    pdf_paths = [p for p in root.rglob("*.pdf")]

    if not pdf_paths:
        print("No PDFs found. Exiting.")
        return

    # 2) gather samples to infer categories
    sample_texts = []
    for pdf in pdf_paths[:5]:
        txt = extract_text_sample(pdf, max_pages=2)
        if txt:
            sample_texts.append(txt)
    sample_texts = sample_texts or ["Sample text unavailable."]

    print("Generating category list from samples...")
    categories = infer_categories(sample_texts, max_cats=5)
    print("Categories inferred:")
    for cat in categories:
        print("  -", cat)

    # 3) classify and move each PDF
    for pdf in pdf_paths:
        txt = extract_text_sample(pdf, max_pages=2)
        cat = classify_document(txt or "No text", categories)
        folder_name = cat.replace(" | ", "--").replace(" ", "_")
        target_dir = root / folder_name
        target_dir.mkdir(exist_ok=True)
        dest = target_dir / pdf.name

        print(f"Moving '{pdf.name}' → '{folder_name}/'")
        try:
            shutil.move(str(pdf), str(dest))
        except Exception as e:
            print(f"  ERROR moving {pdf}: {e}")

    print("Done. PDFs have been reorganised.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organise PDFs into category subfolders using OCI GenAI."
    )
    parser.add_argument(
        "-p", "--path",
        required=True,
        help="Path to the root folder containing PDFs (recursively scanned)"
    )
    args = parser.parse_args()
    main(args.path)
