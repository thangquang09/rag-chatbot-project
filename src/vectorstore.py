import hashlib
import os
from typing import List, Union, Optional, Dict, Any

from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_google_vertexai import VertexAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from loguru import logger

from file_loader import TextSplitter
from setting import K


class QdrantClientManager:
    """Singleton pattern to ensure only one Qdrant client per path to avoid conflicts."""
    _clients: Dict[str, QdrantClient] = {}
    
    @classmethod
    def get_client(cls, path: str) -> QdrantClient:
        """Get or create a Qdrant client for the given path."""
        if path not in cls._clients:
            logger.info(f"Creating new Qdrant client for path: {path}")
            cls._clients[path] = QdrantClient(path=path)
        return cls._clients[path]
    
    @classmethod
    def close_all_clients(cls):
        """Close all clients and clear the cache."""
        for path, client in cls._clients.items():
            try:
                client.close()
                logger.info(f"Closed Qdrant client for path: {path}")
            except Exception as e:
                logger.warning(f"Error closing client for {path}: {e}")
        cls._clients.clear()


class VectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        documents: List[Document] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = VertexAIEmbeddings(
            model="text-embedding-004"
        )
        # Get embedding dimensions for Qdrant configuration
        self.embedding_size = 768  # text-embedding-004 default size
        
        self.sources = set()
        self.vectorstore = None
        self.retriever = None
        self._client = None
        
        # Initialize the vectorstore
        self.vectorstore, self.retriever = self.create_vectorstore(docs_list=documents)

        if documents:
            self._update_sources(documents)

    def _get_client(self) -> QdrantClient:
        """Get the Qdrant client, ensuring only one instance per path."""
        if self._client is None:
            self._client = QdrantClientManager.get_client(self.persist_directory)
        return self._client

    def _update_sources(self, documents: List[Document]):
        """Update list of sources from new documents."""
        for doc in documents:
            if "source" in doc.metadata:
                self.sources.add(doc.metadata["source"])

    def _get_embedding_size(self) -> int:
        """Get the actual embedding size by creating a test embedding."""
        try:
            test_embedding = self.embeddings.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding size, using default 768: {e}")
            return 768

    def _ensure_collection_exists(self, client: QdrantClient):
        """Ensure the collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if not collection_exists:
                # Get actual embedding size
                embedding_size = self._get_embedding_size()
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with embedding size {embedding_size}")
                
                # Create collection with proper vector configuration
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection '{self.collection_name}'")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except ValueError as e:
            if "already exists" in str(e):
                logger.info(f"Collection '{self.collection_name}' already exists")
            else:
                logger.error(f"Error ensuring collection exists: {e}")
                raise
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def check_vectorstore_exists(self) -> bool:
        """Check if the vectorstore already exists."""
        try:
            client = self._get_client()
            collections = client.get_collections()
            return any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
        except Exception as e:
            logger.warning(f"Error checking if vectorstore exists: {e}")
            return False

    def _load_existing_sources(self):
        """Load sources from existing collection."""
        try:
            if self.vectorstore is None:
                return
            
            # Get all points to extract sources
            client = self._get_client()
            scroll_result = client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your needs
                with_payload=True,
                with_vectors=False
            )
            
            self.sources.clear()
            for point in scroll_result[0]:
                if point.payload and "metadata" in point.payload:
                    metadata = point.payload["metadata"]
                    if isinstance(metadata, dict) and "source" in metadata:
                        self.sources.add(metadata["source"])
                        
        except Exception as e:
            logger.warning(f"Error loading existing sources: {e}")

    def create_vectorstore(
        self,
        reload_vectordb: bool = True,
        docs_list: Union[Document, List[Document]] = None,
    ):
        """Create a vectorstore with provided documents or load existing one if reload_vectordb is True."""
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        client = self._get_client()
        vectorstore_exists = self.check_vectorstore_exists()

        # Load existing vectorstore if reload_vectordb is True and vectorstore exists
        if reload_vectordb and vectorstore_exists:
            logger.info("Loading existing Qdrant vector database...")
            try:
                self._ensure_collection_exists(client)
                self.vectorstore = QdrantVectorStore(
                    client=client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                    retrieval_mode=RetrievalMode.DENSE
                )
                
                # Load existing sources
                self._load_existing_sources()
                
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})
                logger.info(f"Loaded existing vectorstore with {len(self.sources)} sources")
                return self.vectorstore, self.retriever
                
            except Exception as e:
                logger.error(f"Error loading existing vectorstore: {e}")
                logger.info("Creating new vectorstore instead...")
                vectorstore_exists = False

        # Create new vectorstore if either reload_vectordb is False or vectorstore doesn't exist
        if reload_vectordb and not vectorstore_exists:
            logger.warning(
                "Reload_vectordb flag is True but no existing vectorstore found. Creating a new one..."
            )

        if docs_list is None:
            # Create a placeholder document if no documents are provided
            logger.info(
                "No documents provided, creating an empty vectorstore with placeholder..."
            )
            placeholder_doc = Document(
                page_content="Placeholder content", metadata={"source": "placeholder"}
            )
            text_splitter = TextSplitter()
            doc_splits = text_splitter(documents=[placeholder_doc])
        else:
            # Process the provided documents
            logger.info(
                f"Creating vectorstore from {len(docs_list) if isinstance(docs_list, list) else 1} document(s)..."
            )
            text_splitter = TextSplitter()
            docs_to_process = docs_list if isinstance(docs_list, list) else [docs_list]
            doc_splits = text_splitter(documents=docs_to_process)

        # Ensure collection exists before creating vectorstore
        self._ensure_collection_exists(client)

        # Create the vectorstore - use client directly instead of from_documents
        self.vectorstore = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE
        )
        
        # Add documents to the vectorstore
        if doc_splits:
            self.vectorstore.add_documents(documents=doc_splits)
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})
        logger.info(f"Created new Qdrant vectorstore with {len(doc_splits)} documents")
        return self.vectorstore, self.retriever

    def add_documents(self, documents: List[Document]):
        """Add pre-split documents to the existing vectorstore, avoiding duplicates.

        Args:
            documents: List of already split/processed Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        
        try:
            client = self._get_client()
            new_docs = []
            new_ids = []
            
            for doc in documents:
                doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
                
                # Check if document already exists
                try:
                    existing_point = client.retrieve(
                        collection_name=self.collection_name,
                        ids=[doc_id]
                    )
                    if not existing_point:
                        new_docs.append(doc)
                        new_ids.append(doc_id)
                except Exception:
                    # If point doesn't exist, add it
                    new_docs.append(doc)
                    new_ids.append(doc_id)
            
            if new_docs:
                self.vectorstore.add_documents(documents=new_docs, ids=new_ids)
                self._update_sources(new_docs)
                logger.info(f"Successfully added {len(new_docs)} new documents to vectorstore")
            else:
                logger.info("No new documents to add; all were duplicates")
                
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {e}")
            raise

    def get_unique_sources(self) -> List[str]:
        """return self.sources"""
        return sorted(list(self.sources)) if self.sources else ["No sources available"]

    def clear_vectorstore(self):
        """Remove all stored documents and keep only a placeholder."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")

        try:
            client = self._get_client()
            
            # Delete the entire collection and recreate it
            try:
                client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'")
            except Exception as e:
                logger.warning(f"Error deleting collection: {e}")
            
            # Recreate the collection
            self._ensure_collection_exists(client)

            # Clear sources
            self.sources.clear()

            # Create new vectorstore instance
            placeholder_doc = Document(
                page_content="Placeholder content", metadata={"source": "placeholder"}
            )
            text_splitter = TextSplitter()
            doc_splits = text_splitter(documents=[placeholder_doc])

            # Recreate vectorstore with placeholder
            self.vectorstore = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                retrieval_mode=RetrievalMode.DENSE
            )
            
            # Add placeholder documents
            if doc_splits:
                self.vectorstore.add_documents(documents=doc_splits)
            
            self._update_sources(doc_splits)
            logger.info("Added placeholder document to vectorstore")

            # Update retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K})

        except Exception as e:
            logger.error(f"Error clearing vectorstore: {e}")
            raise

    def __del__(self):
        """Cleanup method to properly close client connection."""
        try:
            if hasattr(self, '_client') and self._client is not None:
                # Note: We don't close the client here since it might be shared
                # The QdrantClientManager handles client lifecycle
                pass
        except Exception as e:
            logger.warning(f"Error in vectorstore cleanup: {e}")
