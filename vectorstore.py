from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

def get_vectorstore():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
    )
    persist_directory = "chroma_db"
    collection_name = "rag-chroma"

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        logging.info("Loading existing vector database...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
    else:
        logging.info("Creating new vector database...")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=collection_name,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
    
    return vectorstore