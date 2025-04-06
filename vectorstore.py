import logging
import os
from typing import List, Union

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from file_loader import TextSplitter
from setting import K

logging.basicConfig(level=logging.INFO)


class VectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        documents: List[Document] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        self.vectorstore = None
        self.retriever = None
        self.vectorstore, self.retriever = self.create_vectorstore(docs_list=documents)

    def check_vectorstore_exists(self) -> bool:
        """Check if the vectorstore already exists."""
        return os.path.exists(self.persist_directory) and os.listdir(
            self.persist_directory
        )

    def create_vectorstore(
        self,
        reload_vectordb: bool = True,
        docs_list: Union[Document, List[Document]] = None,
    ):
        """Create a vectorstore with provided documents or load existing one if reload_vectordb is True."""
        vectorstore_exists = self.check_vectorstore_exists()

        # Load existing vectorstore if reload_vectordb is True and vectorstore exists
        if reload_vectordb and vectorstore_exists:
            logging.info("Loading existing vector database...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})
            return self.vectorstore, self.retriever

        # Create new vectorstore if either reload_vectordb is False or vectorstore doesn't exist
        if reload_vectordb and not vectorstore_exists:
            logging.warning(
                "Reload_vectordb flag is True but no existing vectorstore found. Creating a new one..."
            )

        if docs_list is None:
            # Create a placeholder document if no documents are provided
            logging.info(
                "No documents provided, creating an empty vectorstore with placeholder..."
            )
            placeholder_doc = Document(
                page_content="Placeholder content", metadata={"source": "placeholder"}
            )
            text_splitter = TextSplitter()
            doc_splits = text_splitter(documents=[placeholder_doc])
        else:
            # Process the provided documents
            logging.info(
                f"Creating vectorstore from {len(docs_list) if isinstance(docs_list, list) else 1} document(s)..."
            )
            text_splitter = TextSplitter()
            docs_to_process = docs_list if isinstance(docs_list, list) else [docs_list]
            doc_splits = text_splitter(documents=docs_to_process)

        # Create the vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        # self.vectorstore.persist()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})
        return self.vectorstore, self.retriever

    def add_documents(self, documents: List[Document]):
        """Add pre-split documents to the existing vectorstore.
    
        Args:
            documents: List of already split/processed Document objects
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vectorstore not initialized. Please create or load it first."
            )

        try:
            # Add documents and persist changes
            self.vectorstore.add_documents(documents)
            # self.vectorstore.persist()
            logging.info(f"Successfully added {len(documents)} documents to vectorstore")
        except Exception as e:
            logging.error(f"Error adding documents to vectorstore: {e}")
            raise


# def get_vectorstore_retriever():
#     urls = [
#         "https://lilianweng.github.io/posts/2023-06-23-agent/",
#         "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#         "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
#     ]

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     persist_directory = "chroma_db"
#     collection_name = "rag-chroma"

#     if os.path.exists(persist_directory) and os.listdir(persist_directory):
#         logging.info("Loading existing vector database...")
#         vectorstore = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embeddings,
#             collection_name=collection_name,
#         )
#     else:
#         logging.info("Creating new vector database...")
#         docs = [WebBaseLoader(url).load() for url in urls]
#         docs_list = [item for sublist in docs for item in sublist]

#         text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
#         )
#         doc_splits = text_splitter.split_documents(docs_list)

#         vectorstore = Chroma.from_documents(
#             documents=doc_splits,
#             collection_name=collection_name,
#             embedding=embeddings,
#             persist_directory=persist_directory,
#         )
#         vectorstore.persist()

#     retriever = vectorstore.as_retriever(search_kwargs={"k": K})
#     return vectorstore, retriever
