from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from urllib.parse import urlparse
import bs4

class WebDataIngestion:
    """
    A class to handle web data ingestion.
    """
    
    def __init__(self, url):
        self.url = url

    def classify_path(self) -> str:
        """
        Classify a given path as either:
        - 'url' if it's an HTTP/HTTPS URL
        - 'local_file' if it's an existing local file
        - 'local_path' if it's a local path that doesn't exist or isn't a file
        """
        parsed = urlparse(self.url)

        if parsed.scheme in ("http", "https") and parsed.netloc:
            return "url"
        elif Path(self.url).is_file():
            return "local_file"
        else:
            return "local_path"

    def fetch_text_data(self):
        """
        Fetches text data from a local file.
        """
        loader = TextLoader(self.url, encoding="utf-8")
        documents = loader.load()
        return documents

    def fetch_web_data(self):
        """
        Fetches web data from the given URL.
        """
        loader = WebBaseLoader(
            web_path=self.url,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            )
        )
        documents = loader.load()
        return documents
    
    def fetch_data(self):
        """
        Fetches data based on the type of URL or file path.
        """
        path_type = self.classify_path()
        
        if path_type == "url":
            return self.fetch_web_data()
        elif path_type == "local_file":
            return self.fetch_text_data()
        else:
            raise ValueError(f"Unsupported path type: {path_type}")

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """
        Splits the fetched documents into smaller chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
if __name__ == "__main__":
    print("Web Data Ingestion Module")
    # url = "https://lilianweng.github.io/posts/2023-06-23-agent/"  # Replace with the actual URL
    url = "/Users/aasharma/Downloads/python_projects/simple_RAG_QnA_chatbot/text_corpse/my_summary.txt"
    
    web_data_ingestion = WebDataIngestion(url)
    documents = web_data_ingestion.fetch_data()
    print(f"Fetched {len(documents)} documents from {url}")
    split_docs = web_data_ingestion.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")