import logging
import os
from typing import List, Union
import hashlib

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
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})
        return self.vectorstore, self.retriever

    def add_documents(self, documents: List[Document]):
        """Add pre-split documents to the existing vectorstore, avoiding duplicates.
    
        Args:
            documents: List of already split/processed Document objects
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vectorstore not initialized. Please create or load it first."
            )

        try:
            # List to hold new documents that aren't duplicates
            new_docs = []
            new_ids = []
            
            for doc in documents:
                # Generate a unique ID based on document content
                doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                # Check if this ID already exists in the vectorstore
                if not self.vectorstore.get(doc_id)["ids"]:
                    new_docs.append(doc)
                    new_ids.append(doc_id)
            
            if new_docs:
                # Add only non-duplicate documents with their IDs
                self.vectorstore.add_documents(documents=new_docs, ids=new_ids)
                logging.info(f"Successfully added {len(new_docs)} new documents to vectorstore")
            else:
                logging.info("No new documents to add; all were duplicates")
        except Exception as e:
            logging.error(f"Error adding documents to vectorstore: {e}")
            raise