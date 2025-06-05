from langchain_chroma import Chroma
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from .prompt import PromptTemplate  

class VectorChromaDB:
    """
    A class to handle interactions with a Chroma vector database.
    """
    def __init__(self, documents: list, embeddings, persist_directory: str):
        """
        Initialize the Chroma vector database with a collection name and persistence directory.
        :param documents: List of documents to be stored in the vector database.
        :param embeddings: Embedding model to be used for the documents.
        :param persist_directory: Directory where the Chroma database is persisted.
        """
        self.documents = documents
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.chroma_client = Chroma.from_documents(
            documents=self.documents, 
            embedding=self.embeddings, 
            persist_directory=self.persist_directory
        )

    # method to return the db as a retriever
    def get_retriever(self, search_kwargs: dict = None):
        """
        Get a retriever from the Chroma vector database.

        :param search_kwargs: Additional search parameters for the retriever.
        :return: A retriever object.
        """
        if search_kwargs is None:
            search_kwargs = {}
        return self.chroma_client.as_retriever(search_kwargs=search_kwargs)
    
    def get_history_aware_retriever(self, llm):
        """
        Get a history-aware retriever from the Chroma vector database.

        :param chat_history: List of previous chat messages.
        :param search_kwargs: Additional search parameters for the retriever.
        :return: A history-aware retriever object.
        """
        
        return create_history_aware_retriever(llm, self.get_retriever(), PromptTemplate().get_contextualize_q_system_prompt())