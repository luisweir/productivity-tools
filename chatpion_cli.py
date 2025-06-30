#!/usr/bin/env python3
# chatpion_cli.py - CLI RAG-enabled OCI Generative AI chatbot with spinner and colored output.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install colorama oci faiss-cpu langchain-community langchain tqdm
#   - Ensure oci.env exists in the same directory with OCI settings
#
# Usage:
#   python chatpion_cli.py [--debug]

import sys
import threading
import time
import itertools
# standard libs for HTML-to-text conversion
import re
import html
import os

from chat_engine import ChatEngine

from colorama import init, Fore, Style

# Initialise colour support
init(autoreset=True)

DEBUG = "--debug" in sys.argv
engine = ChatEngine(debug=DEBUG)

def html_to_text(s: str) -> str:
    """Convert minimal HTML (br, li, tags) to plain-text for CLI."""
    text = re.sub(r'<br\s*/?>', '\n', s)
    text = re.sub(r'</li\s*>', '\n', text)
    text = re.sub(r'<li\s*>', '- ', text)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text)

def show_spinner(stop_event):
    spinner = itertools.cycle(["|", "/", "-", "\\"])
    while not stop_event.is_set():
        print(f"\rThinking... {next(spinner)}", end="", flush=True)
        time.sleep(0.1)
    print("\r", end="", flush=True)

# CLI loop
print(f"{Fore.GREEN}Hi! I am ChatPion CLI. Ask me anything about AI. Type your question or 'exit' to quit.\n")

while True:
    user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    stop_event = threading.Event()
    thread = threading.Thread(target=show_spinner, args=(stop_event,))
    thread.start()

    try:
        answer, docs = engine.chat(user_input)
        stop_event.set()
        thread.join()

        plaintext = html_to_text(answer)
        print(f"\n{Fore.GREEN}ChatPion CLI:{Style.RESET_ALL}\n{plaintext}\n")
        if docs:
            print(f"{Style.DIM}Sources:")
            for idx, doc in enumerate(docs, start=1):
                meta = doc.metadata or {}
                src = meta.get("source", "Unknown source")
                aud = meta.get("audience", "unknown")
                typ = meta.get("type", "unknown")
                print(f"  [{idx}] {src} — {aud} | {typ}")
            print(Style.RESET_ALL)
    except Exception as e:
        stop_event.set()
        thread.join()
        print("Error:", e)