from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def get_current_path():
    return os.path.dirname(os.path.abspath(sys.argv[0]))

persist_directory = f"{get_current_path()}/chroma_langchain_db"

class VectorDB():
    def __init__(
        self,
        persist_directory: str = persist_directory,
    ):
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
        )

        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="langchain",
        )
        
    def add_documents(self, documents):
        """
        Add documents to the vector store.
        """
        self.vectorstore.add_documents(documents)


    def get_retriever(self, k: int = 5):
        """
        Get retriever from the vector store.
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
