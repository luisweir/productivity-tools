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

        sources_html = "<sub>" + "<br>".join(source_lines) + "</sub>"
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
    type="messages"
).launch(server_name="0.0.0.0", server_port=8080)

