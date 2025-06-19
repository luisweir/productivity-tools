#!/usr/bin/env python3
# chatweb.py – Privacy-focused personal assistant to talk to local documents (offline use, without storing knowledge in the internet); powered by OCI Generative AI and FAISS
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install gradio langchain-community faiss-cpu oci
#   - Ensure FAISS index exists by running `python faiss-ingest.py`
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Usage:  python chatweb.py [--debug]

from __future__ import annotations

import os
import re
import signal
import sys
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS

from LoadProperties import LoadProperties

DEBUG = "--debug" in sys.argv

# ─────────────────────────────────────────────────────────────────────────────
# Allowed metadata values
# ─────────────────────────────────────────────────────────────────────────────
ALLOWED_AUDIENCES: dict[str, str] = {
    "business": "Executives, sales, strategy, partners, human resources, people, society…",
    "technical": "Developers, architects, engineers, quality assurance, devops, system design…",
    "general": "Non‑technical and non‑business content intended for a broad audience…",
    "internal": "Oracle internal documents. Assume the user is an Oracle employee.",
}

ALLOWED_TYPES: dict[str, str] = {
    "insight": "Conceptual, strategic thinking or high‑level concepts",
    "deepdive": "Detailed content",
    "research": "Research‑focused publications",
    "governance": "Allowed Oracle tools, internal systems, corporate guidelines, policies and internal processes",
}

# ─────────────────────────────────────────────────────────────────────────────
# OCI clients and FAISS store
# ─────────────────────────────────────────────────────────────────────────────
props = LoadProperties()

llm = ChatOCIGenAI(
    model_id=props.getModelName(),
    service_endpoint=props.getEndpoint(),
    compartment_id=props.getCompartment(),
    model_kwargs={"max_tokens": 800, "temperature": 0.2},
)

embed = OCIGenAIEmbeddings(
    model_id=props.getEmbeddingModelName(),
    service_endpoint=props.getEndpoint(),
    compartment_id=props.getCompartment(),
)

db = FAISS.load_local("faiss_index", embed, allow_dangerous_deserialization=True)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt used to synthesise the answer
# ─────────────────────────────────────────────────────────────────────────────
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful AI assistant. Using the context below, answer the question in a clear, "
        "professional and slightly more elaborate way with HTML formatting.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Classification helper
# ─────────────────────────────────────────────────────────────────────────────

def classify_with_genai(query: str) -> Tuple[str | None, str | None]:
    audience_expl = "\n".join(f"- {k}: {v}" for k, v in ALLOWED_AUDIENCES.items())
    type_expl = "\n".join(f"- {k}: {v}" for k, v in ALLOWED_TYPES.items())

    extra_rules = (
        "Guidelines:\n"
        "• Choose *internal* only if the question explicitly references Oracle internal docs, processes, or tools.\n"
        "• Choose *governance* only for policy, compliance, or review‑process queries.\n"
        "Examples:\n"
        "  Q: 'What does the Oracle AI policy say about model training data?' → internal_governance\n"
        "  Q: 'Explain RAG architecture in simple terms.' → general_insight\n"
        "  Q: 'Show me the Python SDK for OCI Generative AI.' → technical_deepdive\n"
    )

    prompt = (
        "You are an AI classification assistant. Classify the user question into two labels:\n"
        "1. Audience\n2. Type\n\n"
        f"Valid Audience values:\n{audience_expl}\n\n"
        f"Valid Type values:\n{type_expl}\n\n"
        f"{extra_rules}\n"
        "Return only the label in the exact format `audience_type` – no other words.\n\n"
        f"User question:\n{query}\n\nYour response:"
    )

    if DEBUG:
        print("\n[DEBUG] Classification prompt:\n", prompt)

    raw = (
        llm.invoke([HumanMessage(content=prompt)])
        .content.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if DEBUG:
        print("[DEBUG] Classification response:", raw)

    pattern = rf"^({'|'.join(ALLOWED_AUDIENCES)})_({'|'.join(ALLOWED_TYPES)})$"
    m = re.match(pattern, raw)
    return (m.group(1), m.group(2)) if m else (None, None)

# ─────────────────────────────────────────────────────────────────────────────
# Clarification helper
# ─────────────────────────────────────────────────────────────────────────────

def generate_clarifying_prompt(msg: str) -> str:
    prompt = (
        "You're an AI assistant that could not confidently classify the user's intent.\n"
        "Please ask a concise follow‑up question to clarify both audience and type.\n\n"
        f"User message:\n{msg}\n"
    )
    if DEBUG:
        print("[DEBUG] Clarifying prompt input:\n", prompt)
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Main streaming handler
# ─────────────────────────────────────────────────────────────────────────────

def _print_doc_list(docs, label: str = "") -> None:
    if DEBUG:
        print(f"[DEBUG] {label}doc list ({len(docs)} docs):")
        for i, d in enumerate(docs):
            meta = d.metadata or {}
            print(
                f"    • {i}: {Path(meta.get('source', 'Unknown')).name} | "
                f"aud={meta.get('audience')} | type={meta.get('type')}"
            )

def chat_fn_stream(message: str, history: List[dict]):
    if DEBUG:
        print("\n[DEBUG] New user question:", message)

    # Reset
    if message.lower().strip() == "reset session":
        yield "Session reset. Ask away!"
        return

    audience, doc_type = classify_with_genai(message)
    if not audience or not doc_type:
        yield generate_clarifying_prompt(message)
        return

    if DEBUG:
        print(f"[DEBUG] Classified as {audience}_{doc_type}")

    # Filtered search
    filter_dict = {"audience": audience, "type": doc_type}
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5, "filter": filter_dict})
    docs = retriever.invoke(message)

    if docs:
        _print_doc_list(docs, "Filtered ")
    else:
        if DEBUG:
            print("[DEBUG] No docs found with filter", filter_dict)
        # Fallback search
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = retriever.invoke(message)
        _print_doc_list(docs, "Fallback ")

    context_text = "\n\n".join(d.page_content for d in docs)
    answer_prompt = custom_prompt.format(context=context_text, question=message)

    partial = ""
    if hasattr(llm, "stream"):
        for chunk in llm.stream([HumanMessage(content=answer_prompt)]):
            partial += chunk.content
            yield partial.replace("\n", "<br>")
    else:
        partial = llm.invoke([HumanMessage(content=answer_prompt)]).content
        yield partial.replace("\n", "<br>")

    # Build sources list
    seen, items = set(), []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "Unknown")
        if src in seen:
            continue
        seen.add(src)
        fn = Path(src).name
        link = f'<a href="file://{Path(src).resolve()}" target="_blank">{fn}</a>'
        items.append(f"<li>{link} ({meta.get('audience', 'unknown')} | {meta.get('type', 'unknown')})</li>")

    if items:
        sources_html = (
            '<div style="font-size:13px;margin-top:15px;color:#444;">'
            '<strong>Sources used:</strong><ul>' + "".join(items) + '</ul></div>'
        )
        yield partial.replace("\n", "<br>") + sources_html

# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────────────────────

def _shutdown(sig, frame):
    print("\n🛑 Gracefully shutting down…")
    gr.close_all()
    os.kill(os.getpid(), signal.SIGTERM)

signal.signal(signal.SIGINT, _shutdown)

# ─────────────────────────────────────────────────────────────────────────────
# Gradio theme & launch
# ─────────────────────────────────────────────────────────────────────────────

theme = (
    gr.themes.Base(primary_hue="lime", secondary_hue="purple", neutral_hue="gray")
    .set(body_background_fill="linear-gradient(135deg, #f0fff0, #f0f8ff, #fff0f5)")
    .set(chatbot_text_size="17px")
    .set(input_radius="10px")
    .set(button_primary_background_fill="#ffacac")
)

print("🚀 Starting ChatPrompt AI server…")
print("🌐 Access it at: http://localhost:8080")
print("💡 Ask anything related to AI using your Oracle RAG set‑up!")
print("🤔 Powered by Oracle Generative AI")
print("🔚 Press Ctrl + C to shut down the server\n")

gr.ChatInterface(
    fn=chat_fn_stream,
    title="ChatPrompt AI",
    description="Ask anything related to AI. This assistant uses your RAG set‑up with Oracle Generative AI.",
    submit_btn="Ask",
    type="messages",
    theme=theme,
).launch(
    server_name="localhost",
    server_port=8080,
    pwa=True,
    inbrowser=True,
    show_error=True,
)
