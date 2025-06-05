from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

class WebDataIngestion:
    """
    A class to handle web data ingestion.
    """
    
    def __init__(self, url):
        self.url = url

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
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"  # Replace with the actual URL
    
    web_data_ingestion = WebDataIngestion(url)
    documents = web_data_ingestion.fetch_web_data()
    print(f"Fetched {len(documents)} documents from {url}")
    split_docs = web_data_ingestion.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")