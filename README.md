# 🚀 Productivity Tools

Sample Python utilities to help you learn how to use Oracle Cloud Infrastructure (OCI) Generative AI services in hands-on ways. These tools demonstrate techniques for building chat interfaces, indexing documents, classifying content, and summarising audio and video — all designed to support individual learning and experimentation. ✨

## 📑 Table of Contents

- [🚀 Productivity Tools](#-productivity-tools)
  - [📑 Table of Contents](#-table-of-contents)
  - [✅ Prerequisites](#-prerequisites)
  - [⚙️ Configuration](#️-configuration)
  - [🛠️ Setup](#️-setup)
  - [💡 Utilities](#-utilities)
    - [💬 chatpion_web.py](#-chatpion_webpy)
    - [💬 chatpion_cli.py](#-chatpion_cli_py)
    - [🔍 faiss_ingest.py](#-faiss_ingestpy)
    - [📂 classify_docs.py](#-classify_docs_py)
    - [🎤 mic_summary.py](#-mic_summary_py)
    - [🎥 video_summary_gen.py](#-video_summary_genpy)
  - [🤝 Contributing](#-contributing)
  - [License 📜](#license-)
  - [Disclaimer ⚠️](#disclaimer-️)

## ✅ Prerequisites

- Python 3.7 or higher
- pip or pip3
- virtualenv (recommended)
- ffmpeg (macOS: `brew install ffmpeg`; Windows (Chocolatey): `choco install ffmpeg`; or download from https://ffmpeg.org/download.html)
- portaudio (macOS: `brew install portaudio`; Windows: `pip install pipwin && pipwin install pyaudio`)
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

### 💬 chatpion_web.py
**Description:** Privacy-focused personal assistant to talk to local documents (offline use, without storing knowledge in the internet); powered by OCI Generative AI and FAISS
**Prerequisites:**
- Python 3.7 or higher
- pip3 install gradio langchain-community faiss-cpu oci
- Build FAISS index (`python faiss_ingest.py`)
- Ensure OCI CLI config is set up in ~/.oci/config
- Ensure `oci.env` exists in the project root with OCI GenAI and LangChain settings
**Usage:**
```bash
python chatpion_web.py [--debug]
```
Browse to `http://localhost:8080`.

> **Note:** Source document links at the end of each chat are now served via HTTP from the project directory. Ensure your documents are located under the project root so that the generated relative links (served through Gradio's static paths) work correctly in your browser.

### 💬 chatpion_cli.py
**Description:** Boost your productivity with a personal command-line RAG chatbot that retrieves answers from your local data, featuring spinner animation and colored output. Automatically converts HTML-formatted responses into plain text for terminal display.
**Prerequisites:**
- Python 3.7 or higher
- pip3 install colorama oci faiss-cpu langchain-community langchain
- Ensure `oci.env` exists in the project root with OCI GenAI and LangChain settings
**Usage:**
```bash
python chatpion_cli.py [--debug]
```

### 🔍 faiss_ingest.py
**Description:** Boost your productivity by creating a searchable vector store from your PDFs; splits documents into text chunks, generates embeddings with OCI, and indexes them into a FAISS vector store.
**Prerequisites:**
- Python 3.7 or higher
- pip3 install langchain-community oci PyPDF2
- Ensure OCI CLI config is set up in ~/.oci/config
- Recommended: run `./classify_docs.py` to classify the PDFs before ingesting
**Usage:**
```bash
python faiss_ingest.py                       # Load folders from ksources.txt
python faiss_ingest.py --input ./folder1     # Specify one or more folders or files
python faiss_ingest.py --input ./doc.pdf     # Specify a single PDF
python faiss_ingest.py --debug               # Enable verbose logging
python faiss_ingest.py --input ./dir --debug # Combine input and debug options
python faiss_ingest.py --input /path/to/pdf_folder [--debug]
```

### 📂 classify_docs.py
**Description:** Streamline document management by automatically classifying and organizing your PDFs into subfolders using OCI Generative AI.
**Prerequisites:**
- Python 3.7 or higher
- pip3 install langchain-community oci PyPDF2
- Ensure OCI CLI config is set up in ~/.oci/config
**Usage:**
```bash
python classify_docs.py --input /path/to/pdf_folder                    # classify all PDFs in a folder
python classify_docs.py --input file1.pdf file2.pdf /path/to/folder     # classify multiple PDFs or folders
python classify_docs.py --debug --input /path/to/pdf_folder             # enable debug logging
```

### 🎤 mic_summary.py
**Description:** Improve note-taking productivity by recording microphone audio, transcribing speech locally with Whisper, and summarizing the transcript using OCI Generative AI.
**Prerequisites:**
- Python 3.7 or higher
- portaudio (macOS: `brew install portaudio`; Ubuntu: `sudo apt-get install portaudio19-dev`)
- pip3 install openai-whisper sounddevice scipy numpy oci
- Ensure OCI CLI config is set up in ~/.oci/config
**Usage:**
```bash
python mic_summary.py --output-base <base name for output files>
python mic_summary.py --use-transcript <path/to/transcript.txt>
python mic_summary.py --output-dir <directory path>
python mic_summary.py --output-base <base name> --use-transcript <transcript.txt> --output-dir <directory>
```

### 🎥 video_summary_gen.py
**Description:** Accelerate video content analysis by transcribing and summarizing videos; reads video paths from `videos.txt`, extracts audio via Whisper, and generates summaries using OCI Generative AI.
**Prerequisites:**
- Python 3.7 or higher
- ffmpeg (macOS: `brew install ffmpeg`; Ubuntu: `sudo apt-get install ffmpeg`)
- pip3 install openai-whisper oci
- Ensure OCI CLI config is set up in ~/.oci/config
**Usage:**
```bash
python video_summary_gen.py
```
*Videos to process should be listed in `videos.txt`, one video file path per line.*

## 🤝 Contributing

Ideas, feedback, and contributions are welcome. Feel free to open an issue or submit a pull request.

## License 📜

This project is licensed under the [Universal Permissive License v 1.0](https://oss.oracle.com/licenses/upl).

## Disclaimer ⚠️

These scripts are provided for learning and demonstration purposes only. They are not part of any Oracle product or service, and should not be used for commercial, production, or business activities. Outputs may not be reliable and should not be reused beyond personal experimentation. Use is at your own discretion.