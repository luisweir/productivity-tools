#!/usr/bin/env python3
# chatpion-web.py – Privacy‑focused personal assistant to talk to local documents (offline use, no cloud storage) powered by OCI Generative AI.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install gradio langchain-community faiss-cpu oci
#   - Build FAISS index (`python faiss-ingest.py`)
#   - Ensure the OCI CLI config is available in ~/.oci/config
#
# Usage: python chatpion-web.py [--debug]

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import gradio as gr

# dynamic import of shared engine module from dash-named file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "chat_engine", os.path.join(os.path.dirname(__file__), "chat-engine.py")
)
chat_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_engine)
ChatEngine = chat_engine.ChatEngine

DEBUG = "--debug" in sys.argv
engine = ChatEngine(debug=DEBUG)

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
        primary_hue="rose",
        secondary_hue="amber",
        neutral_hue="teal",
    )
    .set(
        body_background_fill="linear-gradient(135deg,#073241 0%,#0d3c49 100%)",
        body_text_color="white",
        body_text_size="15px",

        chatbot_text_size="15px",
        input_text_size="20px",
        input_radius="20px",

        input_background_fill="#0b4a5c",
        input_background_fill_focus="#0d5268",
        input_placeholder_color="#e0e0e0",

        color_accent_soft="#0b4a5c",
        color_accent_soft_dark="#0b4a5c",

        background_fill_primary="#0b4a5c",
        background_fill_primary_dark="#0b4a5c",
        background_fill_secondary="#0a3d50",
        background_fill_secondary_dark="#0a3d50",

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

gr.set_static_paths(paths=[Path(__file__).parent.resolve()])

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

def chat_fn_stream(message: str, history: list[dict]):
    yield from engine.chat_stream(message)

gr.ChatInterface(
    fn=chat_fn_stream,
    title="ChatPion",
    description=(
        "A privacy-centred personal assistant that lets you talk with your local documents. "
        "It runs locally against a local RAG, so document searches never leave your device. "
        "Powered 100% by OCI Gen AI."
    ),
    submit_btn="Talk To Your Data",
    type="messages",
    theme=theme,
    css=CUSTOM_CSS,
).launch(
    server_name="localhost",
    server_port=8080,
    pwa=True,
    inbrowser=True,
    show_error=True,
    allowed_paths=[Path("background.png").resolve()],
)