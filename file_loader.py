import os
import tempfile
from typing import List

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from setting import CHUNK_OVERLAP, CHUNK_SIZE


class TextSplitter:
    def __init__(
        self,
        separators: List[str] = ["\n\n", "\n", ". ", "! ", "? ", ":", ";", " "],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class PDFLoader:
    def __init__(self, text_splitter=None):
        """
        Initialize the PDF loader with an optional text splitter.

        Args:
            text_splitter: An instance of TextSplitter to process documents
        """
        self.text_splitter = text_splitter or TextSplitter()

    def _load_from_file_object(self, pdf_file):
        """
        Helper method to load a PDF from a file object.

        Args:
            pdf_file: A file object from Streamlit's file_uploader

        Returns:
            List[Document]: Raw documents from the PDF
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            temp_path = tmp_file.name

        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        finally:
            # Ensure cleanup happens even if loading fails
            os.unlink(temp_path)

        return documents

    def _load_from_path(self, path_string):
        """
        Helper method to load a PDF from a file path.

        Args:
            path_string: A string path to a PDF file

        Returns:
            List[Document]: Raw documents from the PDF
        """
        loader = PyPDFLoader(path_string)
        return loader.load()

    def load(self, pdf_file=None, path_string=None):
        """
        Load PDF content either from a file uploaded via Streamlit or from a path.

        Args:
            pdf_file: A file object from Streamlit's file_uploader
            path_string: A string path to a PDF file

        Returns:
            List[Document]: A list of document chunks after splitting

        Raises:
            ValueError: If neither pdf_file nor path_string is provided
        """
        if pdf_file is not None:
            documents = self._load_from_file_object(pdf_file)
        elif path_string is not None:
            documents = self._load_from_path(path_string)
        else:
            raise ValueError("Either pdf_file or path_string must be provided")

        # Split the documents into chunks
        return self.text_splitter(documents) if documents else []


class WebLoader:
    def __init__(self, text_splitter=None):
        """
        Initialize the Web loader with an optional text splitter.

        Args:
            text_splitter: An instance of TextSplitter to process documents
        """
        self.text_splitter = text_splitter or TextSplitter()

    def load(self, url):
        """
        Load web content from a URL.

        Args:
            url: The URL to load content from

        Returns:
            List[Document]: A list of document chunks after splitting
        """
        loader = WebBaseLoader(url)
        documents = loader.load()

        # Split the documents into chunks
        return self.text_splitter(documents) if documents else []
