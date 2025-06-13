# RAG Chatbot Project

## Introduction

This is a powerful chatbot application that utilizes the Retrieval-Augmented Generation (RAG) methodology to create context-aware conversational agents. The application combines efficient document retrieval with the power of Large Language Models (LLMs) to provide accurate, contextually relevant answers to user queries.

The core strength of this project lies in its ability to process complex documents, especially PDFs containing both text and tables, through an intelligent parsing and data enrichment pipeline.

## Demo

### User Interface
https://github.com/user-attachments/assets/ce95269b-ab4d-4d7a-83ff-a58a1ca7aecd

### RAG on PDF files and Web content
https://github.com/user-attachments/assets/58395bce-4312-46d1-bd0a-98b880fa23d6

## ⭐ Key Features

This project is equipped with a suite of advanced features that set it apart from standard RAG implementations.

### 🧠 Intelligent RAG Engine

The RAG engine is built using LangGraph, enabling a flexible and self-correcting workflow.

- **Agentic Workflow**: An intelligent agent decides the next best action, such as retrieving documents or responding directly.
- **Document Retrieval**: Employs a retriever to fetch the most relevant documents from a vector database.
- **Relevance Grading**: After retrieval, the system assesses whether the documents are truly relevant to the user's question. If not, it triggers the query rewriting process.
- **Automatic Query Rewriting**: If initial documents are irrelevant, the system can automatically reformulate the user's question to improve the search results' accuracy.
- **Contextual Generation**: The final answer is generated based on a synthesis of information from the retrieved documents and the LLM's base knowledge.

### 📄 Advanced Document Processing with WDMPDFParser

Instead of standard PDF loaders, this project integrates a custom WDMPDFParser for deep analysis and data extraction from PDFs.

- **Multi-Format Support**: Seamlessly processes both PDF files and content from web URLs.
- **Hybrid Extraction**: Extracts both plain text and structured tables from PDF files.
- **Intelligent Separation**: Applies different strategies for text and tables. Text is chunked for optimal retrieval, while tables are kept intact to preserve their structure.
- **Spanned Table Merging**: Automatically detects and merges tables that span across multiple pages into a single, coherent table.
- **Multimodal AI Table Enrichment**:
  - **AI-powered Analysis**: Uses an LLM (Gemini) to analyze the context preceding a table, determine if it has headers, and identify if it marks a new section.
  - **Visual Reconstruction**: Leverages a multimodal model to look at an image of the table alongside the extracted markdown data. This process corrects structural errors and ensures the final table is perfectly formatted and accurate.

### 📊 Table-Specific RAG Enhancements

- **Full Table Context Retrieval**: When a small chunk of a table is retrieved, the system automatically fetches the entire table to provide the LLM with full context. This is crucial for questions requiring calculations or summaries over a whole table.
- **Keyword-based Table Detection**: The system uses configurable keywords to identify data chunks that contain table information, triggering the full-context retrieval feature.

### 💻 User Interface and Experience

The interface is built with Streamlit, focusing on user-friendliness and flexibility.

- **Interactive Chat Interface**: Allows users to easily ask questions and receive answers.
- **Model Selection**: The UI is designed to show options for multiple LLM providers.

**Important Note**: Currently, the advanced document processing features (especially the WDMPDFParser's AI-powered table enrichment) are implemented exclusively with Google Vertex AI. For full functionality, you must configure Google Cloud credentials.

- **Knowledge Base Management**: Easily upload PDFs, add content from URLs, and clear the entire knowledge base with a single click.
- **Credential Management**: An option to provide your Google Cloud credentials is required to unlock the advanced table processing features.
- **Debug Mode**: Toggle a debug mode to view detailed logs of the process, simplifying troubleshooting.

## Prerequisites

- Python 3.10+ installed on your system.
- A Google Cloud account with the Vertex AI API enabled.
- A `.env` file configured with your Google Cloud credentials.

## Configuration

1. Create a file named `.env` in the root directory of the project.

**IMPORTANT**: This project relies on Google Vertex AI for its core features. You must set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the file path of your Google Cloud service account key.

Your `.env` file should look like this:

```bash
# Set the path to your Google Cloud service account JSON key file
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

You can obtain a service account key from the Google Cloud Console by navigating to **IAM & Admin > Service Accounts**. Create a service account, grant it the "Vertex AI User" role, and then create and download a JSON key.

## Installation and Running the App

### Option 1: Using uv (Recommended)

`uv` is an extremely fast Python package installer and resolver, written in Rust.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/thangquang09/rag-chatbot-project.git
   cd rag-chatbot-project
   ```

2. **Install uv**:
   
   See the tutourial on https://docs.astral.sh/uv/getting-started/installation/

3. **Create a Virtual Environment and Install Dependencies**:
   `uv` will create the environment and install packages from `pyproject.toml`/`requirements.txt` in one step.
   ```bash
   uv sync
   ```

4. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate # On macOS/Linux
   .venv\Scripts\activate   # On Windows
   ```

5. **Run the Application**:
   ```bash
   streamlit run src/app.py
   ```

### Option 2: Using pip and venv

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/thangquang09/rag-chatbot-project.git
   cd rag-chatbot-project
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   .venv\Scripts\activate   # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run src/app.py
   ```

### Option 3: Using Docker (Coming soon)

---

**Enjoy using the RAG Chatbot Project!**