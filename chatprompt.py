# chatprompt.py - CLI RAG-enabled Oracle Generative AI chatbot with spinner and colored output.
#
# Prerequisites:
#   - Python 3.7 or higher
#   - pip3 install langchain langchain-community tqdm colorama oci faiss-cpu
#   - Ensure config.txt exists in the same directory with OCI and LangChain settings
#
# Usage:
#   python chatprompt.py

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.prompts import PromptTemplate
from LoadProperties import LoadProperties
from tqdm import tqdm
from colorama import init, Fore, Style
import threading
import time
import itertools

# Initialise colour support
init(autoreset=True)

# Load properties
properties = LoadProperties()

# Set up OCI GenAI LLM
llm = ChatOCIGenAI(
    model_id=properties.getModelName(),
    service_endpoint=properties.getEndpoint(),
    compartment_id=properties.getCompartment(),
    model_kwargs={
        "max_tokens": 600,
        "temperature": 0.5,
        "top_p": 0.95
    }
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

# Define a custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI assistant. Using the context below, answer the question in a clear, professional, and slightly more elaborate way.

Context:
{context}

Question:
{question}

Answer:"""
)

# Create the RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retv,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Spinner animation
def show_spinner(stop_event):
    spinner_cycle = itertools.cycle(["|", "/", "-", "\\"])
    with tqdm(bar_format="Thinking... {desc}", total=1, desc="|") as pbar:
        while not stop_event.is_set():
            pbar.set_description(next(spinner_cycle))
            time.sleep(0.1)

# CLI loop
print(f"{Fore.GREEN}Hi! I am ChatPrompt. Ask me anything about AI. Type your question or 'exit' to quit.\n")

while True:
    user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=show_spinner, args=(stop_spinner,))
    spinner_thread.start()

    try:
        response = chain.invoke(user_input)
        stop_spinner.set()
        spinner_thread.join()

        print(f"\n{Fore.GREEN}ChatPrompt:{Style.RESET_ALL}\n{response['result']}\n")

        print(f"{Style.DIM}Sources:")
        for i, doc in enumerate(response["source_documents"]):
            source = doc.metadata.get("source", "Unknown source")
            audience = doc.metadata.get("audience", "unknown")
            domain = doc.metadata.get("domain", "unknown")
            doc_type = doc.metadata.get("type", "unknown")
            print(f"  [{i+1}] {source} — {audience} | {domain} | {doc_type}")
        print(Style.RESET_ALL)

    except Exception as e:
        stop_spinner.set()
        spinner_thread.join()
        print("Error:", str(e))
