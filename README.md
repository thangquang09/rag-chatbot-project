# RAG Chatbot Project

RAG Chatbot Project is a powerful application that uses Retrieval-Augmented Generation (RAG) to create context-aware chatbots. The application combines document retrieval with language model capabilities to provide accurate, contextually relevant responses to user queries.

## Features
- Process and index documents for knowledge retrieval
- Generate context-aware responses using RAG methodology
- Support for various document formats
- Interactive chat interface

## Prerequisites
1. Python 3.x installed on your system (3.12)
2. Docker installed if you prefer running the application in a container
3. A `.env` file containing your API keys for LLM integration

### Example `.env` file:
```
GOOGLE_API_KEY=your_key_here
```

---

## Running the Application

### Option 1: Using Python Virtual Environment

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/thangquang09/rag-chatbot-project.git
    cd rag-chatbot-project
    ```

2. **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

---

### Option 2: Using Docker (Comming soon)

---

Enjoy using RAG Chatbot Project!