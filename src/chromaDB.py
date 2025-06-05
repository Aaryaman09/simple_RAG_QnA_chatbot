from langchain_chroma import Chroma

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


if __name__ == "__main__":
    # huggingface embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    

    # Sample documents and embeddings
    # convert sample list of text into documents using document class
    from langchain_core.documents import Document
    sample_documents = [Document(page_content="This is a sample document."), Document(page_content="This is another document.")]

    # Initialize the Chroma vector database
    vector_db = VectorChromaDB(
        documents=sample_documents, 
        embeddings=embeddings, 
        persist_directory="./chroma_db"
    )

    # Get the retriever
    retriever = vector_db.get_retriever()
    print(retriever)
    print("Retriever initialized successfully.")