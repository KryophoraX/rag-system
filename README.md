# RAG System

A lightweight Retrieval-Augmented Generation (RAG) system built with **LangChain**, **NVIDIA NIM**, **Hugging Face embeddings**, and **FAISS**.

The system loads a PDF document, converts it into searchable chunks, stores vector embeddings locally, retrieves the most relevant sections for a question, and uses an NVIDIA-hosted LLM to generate an answer based only on the retrieved document context.

## Architecture

```text
PDF Document
     │
     ▼
PyPDFLoader
     │
     ▼
Text Chunking
     │
     ▼
Hugging Face Embeddings
     │
     ▼
FAISS Vector Database
     │
     ▼
Similarity Search
     │
     ▼
Relevant Context
     │
     ▼
NVIDIA NIM LLM
     │
     ▼
Answer + Sources
```

## Features

* 📄 PDF document ingestion
* 🔎 Semantic similarity search
* 🧠 Local Hugging Face embeddings
* ⚡ FAISS vector database
* 🤖 NVIDIA NIM for LLM inference
* 🔗 LangChain-based RAG pipeline
* 📚 Source/page references in responses
* 🔐 API keys stored securely in `.env`
* 🚫 Answers are restricted to retrieved document context

## Tech Stack

| Component       | Technology                               |
| --------------- | ---------------------------------------- |
| LLM             | NVIDIA NIM                               |
| RAG Framework   | LangChain                                |
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | FAISS                                    |
| PDF Processing  | PyPDF                                    |
| Language        | Python                                   |

## Project Structure

```text
rag-system/
├── .env
├── .gitignore
├── ask.py
├── ingest.py
├── requirements.txt
├── documents/
│   └── your_file.pdf
├── faiss_index/
└── .venv/
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-system
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If your shell has a `python` alias that overrides the virtual environment, use:

```bash
unalias python pip
source .venv/bin/activate
```

Verify:

```bash
which python
```

It should point to:

```text
.../rag-system/.venv/bin/python
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## NVIDIA API Key

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Do **not** commit your API key to GitHub.

The `.gitignore` file already excludes `.env`.

## Add a Document

Place your PDF inside:

```text
documents/
```

Then update the PDF path in `ingest.py`:

```python
PDF_PATH = "documents/your_file.pdf"
```

## Build the Vector Database

Run:

```bash
python ingest.py
```

You should see output similar to:

```text
Loaded X pages
Created X chunks
FAISS database created successfully!
```

This creates a local:

```text
faiss_index/
```

directory containing the vector database.

## Ask Questions

Run:

```bash
python ask.py
```

The program will prompt:

```text
Ask a question:
```

Enter a question about the document.

The system retrieves the three most relevant chunks and sends them to the NVIDIA model along with the question.

The response includes:

```text
Answer:
...

Sources:
- documents/your_file.pdf, page X
- documents/your_file.pdf, page Y
- documents/your_file.pdf, page Z
```

## How RAG Works

Instead of sending the entire document to the LLM, this project uses Retrieval-Augmented Generation.

### 1. Ingestion

The PDF is loaded using `PyPDFLoader`.

### 2. Chunking

The document is divided into smaller sections using `RecursiveCharacterTextSplitter`.

The current configuration uses:

```python
chunk_size=1000
chunk_overlap=200
```

### 3. Embeddings

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model runs locally.

### 4. Vector Search

FAISS stores the embeddings and performs similarity searches when a user asks a question.

### 5. Generation

The most relevant chunks are provided as context to the NVIDIA NIM model.

The model is instructed to answer using only the retrieved context.

## Security

Never commit secrets such as:

```text
NVIDIA_API_KEY
```

The `.env` file is intentionally excluded from Git.

If an API key is accidentally exposed publicly, revoke it and generate a new one.

## Current Limitations

* The system currently processes PDF documents.
* Retrieval is limited to the top 3 matching chunks.
* The quality of answers depends on document quality and chunk retrieval.
* Embeddings run locally, while LLM inference uses NVIDIA's hosted API.
* The FAISS database must be regenerated when documents change.

## Future Improvements

Potential improvements include:

* Multiple-document support
* Automatic document discovery
* Better citation formatting
* Metadata filtering
* Conversational memory
* Streaming responses
* Reranking retrieved chunks
* Hybrid keyword + semantic search
* Web-based chat interface
* Persistent document management
* Support for additional file formats

## License

This project is intended for educational and experimental use.
