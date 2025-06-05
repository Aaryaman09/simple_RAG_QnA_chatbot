from langchain_huggingface import HuggingFaceEmbeddings

# create a class for HGFembedding that initializes the HuggingFaceEmbeddings and returns the embeddings
class HGFembedding:
    """
    A class to create HuggingFace embeddings.
    This class initializes the HuggingFaceEmbeddings with a specified model name.
    """   
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the HuggingFaceEmbeddings with the specified model name.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def get_embeddings(self):
        """
        Returns the initialized HuggingFace embeddings.
        """
        return self.embeddings