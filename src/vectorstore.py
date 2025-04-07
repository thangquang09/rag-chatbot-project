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
        self.sources = set()
        self.vectorstore = None
        self.retriever = None
        self.vectorstore, self.retriever = self.create_vectorstore(docs_list=documents)
        
        if documents:
            self._update_sources(documents)
            
    def _update_sources(self, documents: List[Document]):
        """Update list of sources from new documents."""
        for doc in documents:
            if "source" in doc.metadata:
                self.sources.add(doc.metadata["source"])
        
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
            all_docs = self.vectorstore.get()
            self.sources.clear()
            for metadata in all_docs.get("metadatas", []):
                if metadata and "source" in metadata:
                    self.sources.add(metadata["source"])
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
            raise ValueError("Vectorstore not initialized.")
        try:
            new_docs = []
            new_ids = []
            for doc in documents:
                doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                if not self.vectorstore.get(doc_id)["ids"]:
                    new_docs.append(doc)
                    new_ids.append(doc_id)
            if new_docs:
                self.vectorstore.add_documents(documents=new_docs, ids=new_ids)
                self._update_sources(new_docs)  # update sources when new documents are added
                logging.info(f"Successfully added {len(new_docs)} new documents to vectorstore")
            else:
                logging.info("No new documents to add; all were duplicates")
        except Exception as e:
            logging.error(f"Error adding documents to vectorstore: {e}")
            raise
        
    def get_unique_sources(self) -> List[str]:
        """return self.sources"""
        return sorted(list(self.sources)) if self.sources else ["No sources available"]
    
    def clear_vectorstore(self):
        """Remove all stored documents and keep only a placeholder."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        
        try:
            # Delete all documents from the vectorstore
            all_ids = self.vectorstore.get()["ids"]
            if all_ids:
                self.vectorstore.delete(ids=all_ids)
                logging.info(f"Deleted {len(all_ids)} documents from vectorstore")
            
            # Delete sources
            self.sources.clear()
            
            # Add placeholder document
            placeholder_doc = Document(
                page_content="Placeholder content",
                metadata={"source": "placeholder"}
            )
            text_splitter = TextSplitter()
            doc_splits = text_splitter(documents=[placeholder_doc])
            
            # Add placeholder into vectorstore
            self.vectorstore.add_documents(documents=doc_splits)
            self._update_sources(doc_splits)
            logging.info("Added placeholder document to vectorstore")
            
            # Update retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})
            
        except Exception as e:
            logging.error(f"Error clearing vectorstore: {e}")
            raise