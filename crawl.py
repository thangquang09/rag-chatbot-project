import logging
from typing import List

import validators
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from setting import CHUNK_OVERLAP, CHUNK_SIZE, USER_AGENT

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_valid_url(url: str) -> bool:
    """
    Check if the provided URL is valid.
    """
    if not validators.url(url):
        logging.error("Invalid URL provided.")
        return False

    return True


def load_web_content(url: str) -> List[Document]:
    try:
        loader = WebBaseLoader(
            web_paths=(url,), header_template={"User-Agent": USER_AGENT}
        )

        documents = loader.load()
        if not documents:
            raise ValueError("Can't load content from URL.")

        # Documents Preprocessing
        for doc in documents:
            doc.page_content = " ".join(doc.page_content.split())
            doc.page_content = doc.page_content.strip()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        all_splits = text_splitter.split_documents(documents)

        return all_splits
    except Exception as e:
        raise Exception(f"Error loading documents: {str(e)}")
