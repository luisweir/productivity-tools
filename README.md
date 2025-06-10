# 🚀 Productivity Tools

Sample Python utilities demonstrating how to leverage Oracle Cloud Infrastructure (OCI) Generative AI services to boost productivity across common tasks such as chat interfaces, document indexing, classification, and multimedia summarization. ✨

## 📑 Table of Contents

- [🚀 Productivity Tools](#-productivity-tools)
  - [📑 Table of Contents](#-table-of-contents)
  - [✅ Prerequisites](#-prerequisites)
  - [⚙️ Configuration](#️-configuration)
  - [🛠️ Setup](#️-setup)
  - [💡 Utilities](#-utilities)
    - [💬 chatweb.py](#-chatwebpy)
    - [💬 chatprompt.py](#-chatpromptpy)
    - [🔍 faiss-ingest.py](#-faiss-ingestpy)
    - [📂 organise\_pdfs.py](#-organise_pdfspy)
    - [🎤 mic-summary.py](#-mic-summarypy)
    - [🎥 video-summary-oci.py](#-video-summary-ocipy)
  - [🤝 Contributing](#-contributing)
  - [License 📜](#license-)
  - [Disclaimer ⚠️](#disclaimer-️)

## ✅ Prerequisites

- Python 3.7 or higher
- pip or pip3
- virtualenv (recommended)
- ffmpeg (macOS: `brew install ffmpeg`; Ubuntu: `sudo apt-get install ffmpeg`)
- portaudio (macOS: `brew install portaudio`; Ubuntu: `sudo apt-get install portaudio19-dev`)
- [OCI CLI](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm) configured (~/.oci/config)
- `oci.env` file with OCI Gen AI and LangChain settings (see [Configuration](#configuration))

## ⚙️ Configuration

Create or update the `oci.env` file in the project root with your OCI Gen AI and LangChain settings:

```json
{
  "default_profile": "DEFAULT",
  "model_name": "<OCI Gen AI model name>",
  "model_ocid": "<OCI Gen AI model OCID>",
  "embedding_model_name": "<OCI embedding model name>",
  "endpoint": "<OCI Gen AI endpoint URL>",
  "compartment_ocid": "<OCI compartment OCID>",
  "langchain_endpoint": "<LangChain API endpoint>",
  "langchain_key": "<LangChain API key>"
}
```

## 🛠️ Setup

1. Create and activate a virtual environment:

   ```bash
   pip install virtualenv
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   _If no `requirements.txt` is provided, install dependencies manually (e.g., `pip install oci openai langchain faiss-cpu whisper gradio tkinter`)._

## 💡 Utilities

### 💬 chatweb.py
**Description:** Launches a Gradio web interface that serves a Retrieval-Augmented Generation (RAG) chatbot powered by OCI Generative AI and a FAISS vector store.
**Setup:** Ensure you have built the FAISS index by running `faiss-ingest.py`.
**Usage:**
```bash
python chatweb.py
```

Browse to `http://localhost:8080`.

### 💬 chatprompt.py
**Description:** Interactive command-line RAG chatbot with spinner animation and colored output.
**Usage:**
```bash
python chatprompt.py
```

### 🔍 faiss-ingest.py
**Description:** Splits PDF documents into text chunks, generates embeddings with OCI, and indexes them into a FAISS vector store.
**Before Running:** Update the `pdf_dirs` list at the top of the script to point to your PDF directories.
**Usage:**
```bash
python faiss-ingest.py
```

### 📂 organise_pdfs.py
**Description:** Scans a folder for PDF files, classifies each using OCI Generative AI, and moves them into categorized subfolders.
**Usage:**
```bash
python organise_pdfs.py -p /path/to/pdf_folder
```

### 🎤 mic-summary.py
**Description:** Records audio from the microphone, transcribes speech locally using Whisper, and summarizes the transcript with OCI Generative AI.
**Usage:**
```bash
python mic-summary.py --output-base <base_name>
```

### 🎥 video-summary-gen.py
**Description:** Reads a list of video file paths from `videos.txt`, extracts and transcribes audio via Whisper, and generates summaries using OCI Generative AI. Outputs a summary and full transcript file for each video.
**Usage:**
```bash
python video-summary-gen.py
```
_Videos to process should be listed in `videos.txt`, one video file path per line._

## 🤝 Contributing

Contributions and feedback are welcome! Please open issues or pull requests to enhance these utilities.

## License 📜

This project is available under the [Universal Permissive License v 1.0](https://oss.oracle.com/licenses/upl).

See [LICENSE](LICENSE.txt) for details.

## Disclaimer ⚠️

This is **NOT** an official Oracle product. It is provided for demonstration purposes only, without any guarantees of reliability, accuracy, or completeness. Use it at your own risk.