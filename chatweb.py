#!/usr/bin/env python3
# chatweb.py – Privacy‑focused personal assistant to talk to local documents (offline use, no cloud storage) powered by OCI Generative AI and FAISS
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install gradio langchain-community faiss-cpu oci
#   - Build the FAISS index first with `python faiss-ingest.py`
#   - Ensure the OCI CLI config is available in ~/.oci/config
#
# Usage:  python chatweb.py [--debug]

from __future__ import annotations

import os
import re
import signal
import sys
from collections import OrderedDict
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
    "governance": "Internal tools, corporate guidelines, policies and processes",
}

# ─────────────────────────────────────────────────────────────────────────────
# OCI clients and FAISS store
# ─────────────────────────────────────────────────────────────────────────────
props = LoadProperties()

llm = ChatOCIGenAI(
    model_id=props.getModelName(),
    service_endpoint=props.getEndpoint(),
    compartment_id=props.getCompartment(),
    model_kwargs={
        # allows concise yet complete answers
        "max_tokens": 500,

        # slight randomness
        "temperature": 0.25,     # 0.2-0.4 is usually the sweet spot

        # nucleus sampling keeps surprises rare but possible
        "top_p": 0.85,           # >0.8 keeps response grounded

        # modest top-k so off-topic words rarely enter
        "top_k": 40,             # 20-60 works well

        # mild repetition control
        "frequency_penalty": 0.1,
        "presence_penalty": 0.0,

        # one variant is enough for RAG
        "num_generations": 1,
    },
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
        "• Choose *internal* only if the question explicitly references Oracle internal docs, processes or tools.\n"
        "• Choose *governance* only for policy, compliance or review‑process queries.\n"
        "Examples:\n"
        "  Q: 'What does the Oracle AI policy say about model training data?' → internal_governance\n"
        "  Q: 'Explain RAG architecture in simple terms.' → general_insight\n"
        "  Q: 'Show me the Python SDK for OCI Generative AI.' → technical_deepdive\n"
    )

    prompt = (
        "You are an AI classification assistant. Classify the user question into two labels (audience and type).\n\n"
        f"Valid Audience values:\n{audience_expl}\n\n"
        f"Valid Type values:\n{type_expl}\n\n"
        f"{extra_rules}\n"
        "Return the labels in the exact format `audience_type` with no other text.\n\n"
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
        "Ask one concise follow‑up question to clarify both audience and type.\n\n"
        f"User message:\n{msg}\n"
    )
    if DEBUG:
        print("[DEBUG] Clarifying prompt input:\n", prompt)
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Ranked retrieval across all audience/type combos
# ─────────────────────────────────────────────────────────────────────────────

def ranked_retrieval(query: str, audience: str, doc_type: str, k: int = 5) -> List:
    """Retrieve up to *k* docs, giving priority to those whose metadata matches the predicted audience and type."""

    results: list[tuple[int, object]] = []

    aud_rank = [audience] + [a for a in ALLOWED_AUDIENCES if a != audience]
    typ_rank = [doc_type] + [t for t in ALLOWED_TYPES if t != doc_type]

    for a_idx, aud in enumerate(aud_rank):
        for t_idx, typ in enumerate(typ_rank):
            filter_dict = {"audience": aud, "type": typ}
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k, "filter": filter_dict},
            )
            for d in retriever.invoke(query):
                score = a_idx + t_idx  # lower score = better match
                results.append((score, d))

    best: "OrderedDict[str, tuple[int, object]]" = OrderedDict()
    for score, doc in results:
        key = doc.metadata.get("source", "") + "#" + str(doc.metadata.get("page", 0))
        if key not in best or score < best[key][0]:
            best[key] = (score, doc)

    ordered = sorted(best.values(), key=lambda x: x[0])[:k]
    return [d for _score, d in ordered]

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
    gr.themes.Base(
        primary_hue="rose",      # coral
        secondary_hue="amber",   # gold
        neutral_hue="teal",      # deep teal
    )
    .set(
        # page / canvas
        body_background_fill="linear-gradient(135deg,#073241 0%,#0d3c49 100%)",
        body_text_color="white",
        body_text_size="15px",

        # chat sizes
        chatbot_text_size="15px",
        input_text_size="20px",
        input_radius="20px",

        # input field
        input_background_fill="#0b4a5c",
        input_background_fill_focus="#0d5268",
        input_placeholder_color="#e0e0e0",

        # user bubble (accent) – dark teal
        color_accent_soft="#0b4a5c",
        color_accent_soft_dark="#0b4a5c",

        # bot bubble (primary) – dark teal
        background_fill_primary="#0b4a5c",
        background_fill_primary_dark="#0b4a5c",
        background_fill_secondary="#0a3d50",
        background_fill_secondary_dark="#0a3d50",

        # bigger buttons
        button_large_padding="14px 36px",
        button_large_text_size="18px",
        button_large_radius="24px",

        button_primary_background_fill="#ff8c8c",
        button_primary_text_color="white",
        button_secondary_background_fill="#e6c36f",
        button_secondary_text_color="#062e3a",

        border_color_primary="#e6c36f",
        shadow_drop="0 4px 12px rgba(0,0,0,0.25)",
    )
)

gr.set_static_paths(paths=[Path(__file__).parent.resolve()])   # make local files public

CUSTOM_CSS = """
.gradio-container {
    background: url('/gradio_api/file=background.png') center / cover no-repeat fixed;
}
"""

print("🚀 Starting ChatPrompt AI server…")
print("🌐 Access it at: http://localhost:8080")
print("💡 Ask anything related to AI using your Oracle RAG set‑up!")
print("🤔 Powered by Oracle Generative AI")
print("🔚 Press Ctrl + C to shut down the server\n")

# … later, when you build the UI …
gr.ChatInterface(
    fn=chat_fn_stream,
    title="ChatPion",
    description="A privacy-centred personal assistant that lets you talk with your local documents. It runs a locally against a local RAG, so document searches don't ever leaves your device. Powered by OCI Generative AI.",
    submit_btn="Talk To Your Data",
    type="messages",
    theme=theme,
    css=CUSTOM_CSS,                      # <- new line
).launch(
    server_name="localhost",
    server_port=8080,
    pwa=True,
    inbrowser=True,
    show_error=True,
    allowed_paths=[Path("background.png").resolve()],          # <- new line
)
