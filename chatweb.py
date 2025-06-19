#!/usr/bin/env python3
# chatweb.py - Launch a Gradio web UI for RAG-based chat using OCI Generative AI and FAISS vector store.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install gradio langchain-community faiss-cpu oci
#   - Ensure FAISS index exists by running `python faiss-ingest.py`
#   - Ensure OCI CLI config is set up in ~/.oci/config
#
# Usage:
#   python chatweb.py

import gradio as gr
import os
import signal
import time

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.prompts import PromptTemplate
from LoadProperties import LoadProperties

# Load properties
properties = LoadProperties()

# Define theme
pastel_oasis_theme = gr.themes.Base(
    primary_hue="lime",
    secondary_hue="purple",
    neutral_hue="gray",
).set(
    body_background_fill="linear-gradient(135deg, #f0fff0, #f0f8ff, #fff0f5)",
    chatbot_text_size="17px",
    input_radius="10px",
    button_primary_background_fill="#ffacac",
)

# Set up LLM
llm = ChatOCIGenAI(
    model_id=properties.getModelName(),
    service_endpoint=properties.getEndpoint(),
    compartment_id=properties.getCompartment(),
    model_kwargs={"max_tokens": 800, "temperature": 0.5}
)

# Set up embeddings
embeddings = OCIGenAIEmbeddings(
    model_id=properties.getEmbeddingModelName(),
    service_endpoint=properties.getEndpoint(),
    compartment_id=properties.getCompartment(),
)

# Load FAISS index
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI assistant. Using the context below, answer the question in a clear, professional, and slightly more elaborate way.

Context:
{context}

Question:
{question}

Answer:"""
)

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retv,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Simulated streaming chat function
def chat_fn_stream(message, chat_history):
    try:
        response = qa_chain.invoke(message)
        answer = response["result"]

        partial = ""
        for word in answer.split():
            partial += word + " "
            time.sleep(0.03)
            yield chat_history + [{"role": "assistant", "content": partial}]

        sources = response["source_documents"]
        source_lines = []
        for i, doc in enumerate(sources):
            source = doc.metadata.get("source", "Unknown")
            audience = doc.metadata.get("audience", "unknown")
            doc_type = doc.metadata.get("type", "unknown")
            abs_path = os.path.abspath(source)
            file_name = os.path.basename(source)
            file_link = f'<a href="file://{abs_path}" target="_blank">{file_name}</a>'
            meta = f"{audience} | {doc_type}"
            source_lines.append(f"[{i+1}] {file_link} — {meta}")

        sources_html = (
            '<div style="font-size: 12px; color: #555; margin-top: 10px;">'
            + "<br>".join(source_lines)
            + "</div>"
        )
        final_output = partial + "<br><br>" + sources_html
        yield chat_history + [{"role": "assistant", "content": final_output}]
    except Exception as e:
        yield chat_history + [{"role": "assistant", "content": f"Error: {str(e)}"}]

# Graceful shutdown handler
def shutdown_handler(sig, frame):
    print("\n🛑 Gracefully shutting down...")
    gr.close_all()
    os.kill(os.getpid(), signal.SIGTERM)

signal.signal(signal.SIGINT, shutdown_handler)

# Print startup log
print("🚀 Starting ChatPrompt AI server...")
print("🌐 Access it at: http://localhost:8080")
print("💡 Ask anything related to AI using your Oracle RAG setup!")
print("📁 Local file links will open in your default file viewer (if supported).")
print("🧠 Powered by Oracle Generative AI")
print("🔚 \033[90mPress Ctrl + C to shut down the server\033[0m\n")

# Launch the UI
gr.ChatInterface(
    fn=chat_fn_stream,
    title="ChatPrompt AI",
    description="Ask anything related to AI. This assistant uses your RAG setup with Oracle Generative AI.",
    submit_btn="Ask",
    type="messages",
    theme=pastel_oasis_theme
).launch(
    server_name="localhost",
    server_port=8080,
    pwa=True,
    inbrowser=True,
    show_error=True
)