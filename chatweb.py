import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.prompts import PromptTemplate
from LoadProperties import LoadProperties
import os

# Load properties
properties = LoadProperties()

pastel_oasis_theme = gr.themes.Base(
    primary_hue="lime",
    secondary_hue="purple",
    neutral_hue="gray",
).set(
    # Body
    body_background_fill="linear-gradient(135deg, #f0fff0, #f0f8ff, #fff0f5)",
    body_background_fill_dark="linear-gradient(135deg, #2c3e50, #34495e)",
    body_text_color="#4f4f4f",
    body_text_color_dark="#ffffff",
    body_text_size="17px",

    # Element Colors
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#1f2937",
    background_fill_secondary="#f7f7f7",
    background_fill_secondary_dark="#374151",
    border_color_accent="#ffdab9", # Peach
    border_color_accent_dark="#ffa07a", # Light Salmon
    border_color_primary="#e6e6e6",
    border_color_primary_dark="#4b5563",
    color_accent="#98d8a8", # Pastel Green
    color_accent_soft="#e0f0e3",
    color_accent_soft_dark="#228b22", # Forest Green

    # Text
    link_text_color="#8a9afc", # Soft Lavender
    link_text_color_dark="#b3befe",
    link_text_color_active="#697dfc",
    link_text_color_active_dark="#8a9afc",
    link_text_color_hover="#697dfc",
    link_text_color_hover_dark="#8a9afc",
    prose_text_size="17px",
    prose_text_weight="400",
    code_background_fill="#f5f5f5",
    code_background_fill_dark="#2d2d2d",

    # Shadows
    shadow_drop="rgba(0, 0, 0, 0.05) 0px 1px 2px 0px",
    shadow_drop_lg="rgba(0, 0, 0, 0.08) 0px 4px 12px 0px",
    shadow_inset="rgba(0, 0, 0, 0.05) 0px 2px 4px 0px inset",

    # Layout
    block_background_fill="#ffffff",
    block_background_fill_dark="#1f2937",
    block_border_color="#e6e6e6",
    block_border_color_dark="#4b5563",
    block_border_width="1px",
    block_label_background_fill="#98d8a8", # Pastel Green
    block_label_background_fill_dark="#228b22",
    block_label_text_color="#4f4f4f",
    block_label_text_color_dark="#ffffff",
    block_radius="12px",
    block_shadow="0px 1px 3px rgba(0,0,0,0.05)",
    block_title_text_weight="600",
    container_radius="16px",
    layout_gap="20px",

    # Component Atoms
    chatbot_text_size="17px",
    input_background_fill="#f9f9f9",
    input_background_fill_dark="#2d3748",
    input_border_color="#dcdcdc",
    input_border_color_dark="#4a5568",
    input_border_color_focus="#ffc0cb", # Pastel Pink
    input_border_color_focus_dark="#ff7f9e",
    input_placeholder_color="#a9a9a9",
    input_placeholder_color_dark="#9ca3af",
    input_radius="10px",
    input_shadow="none",
    input_shadow_focus="0 0 0 2px #ffc0cb50",
    input_text_size="17px",

    # Buttons
    button_border_width="1px",
    button_large_radius="12px",
    button_large_text_weight="600", # This correctly styles large buttons
    button_small_radius="8px",
    button_transition="background-color 0.2s ease-in-out, border-color 0.2s ease-in-out, color 0.2s ease-in-out",

    # Primary Buttons
    button_primary_background_fill="#ffacac", # Pastel Coral
    button_primary_background_fill_dark="#ff8c8c",
    button_primary_background_fill_hover="#ff9292",
    button_primary_background_fill_hover_dark="#ff7878",
    button_primary_border_color="#ffacac",
    button_primary_border_color_dark="#ff8c8c",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",

    # Secondary Buttons
    button_secondary_background_fill="#f0e6ff", # Light Lavender
    button_secondary_background_fill_dark="#4c3b6e",
    button_secondary_background_fill_hover="#e6d9ff",
    button_secondary_background_fill_hover_dark="#5a4682",
    button_secondary_border_color="#dcd0ff",
    button_secondary_border_color_dark="#4c3b6e",
    button_secondary_text_color="#5f4b8b", # Dark Lavender
    button_secondary_text_color_dark="#ffffff",
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

def chat_fn(message, chat_history):
    try:
        response = qa_chain.invoke(message)
        answer = response["result"]

        # Format sources with smaller font and HTML links
        sources = response["source_documents"]
        source_lines = []
        for i, doc in enumerate(sources):
            source = doc.metadata.get("source", "Unknown")
            audience = doc.metadata.get("audience", "unknown")
            domain = doc.metadata.get("domain", "unknown")
            doc_type = doc.metadata.get("type", "unknown")

            abs_path = os.path.abspath(source)
            file_name = os.path.basename(source)
            file_link = f'<a href="file://{abs_path}" target="_blank">{file_name}</a>'
            meta = f"{audience} | {domain} | {doc_type}"

            source_lines.append(f"[{i+1}] {file_link} — {meta}")

        sources_html = (
            '<div style="font-size: 12px; color: #555; margin-top: 10px;">'
            + "<br>".join(source_lines)
            + "</div>"
        )
        final_output = answer + "<br><br>" + sources_html

        return chat_history + [{"role": "assistant", "content": final_output}]
    except Exception as e:
        return chat_history + [{"role": "assistant", "content": f"Error: {str(e)}"}]

# Launch Gradio chat with startup log
print("🚀 Starting ChatPrompt AI server...")
print("🌐 Access it at: http://localhost:8080")
print("💡 Ask anything related to AI using your Oracle RAG setup!")
print("📁 Local file links will open in your default file viewer (if supported).")
print("🧠 Powered by Oracle Generative AI\n")

gr.ChatInterface(
    fn=chat_fn,
    title="ChatPrompt AI",
    description="Ask anything related to AI. This assistant uses your RAG setup with Oracle Generative AI.",
    submit_btn="Ask",
    type="messages",
    theme=pastel_oasis_theme
).launch(server_name="0.0.0.0", server_port=8080)

