
from langchain_ollama import ChatOllama as Ollama
from langchain_groq import ChatGroq

class FetchLLM:
    """A class to fetch and initialize the appropriate LLM model based on the service type."""
    def __init__(self, llm_service: str, model_name:str, groq_api_key: str = ""):
        """
        Initializes the FetchLLM class with the specified LLM service, model name, and Groq API key(optional).
        """
        self.llm_service = llm_service
        self.model_name = model_name
        self.groq_api_key = groq_api_key

    def get_model(self):
        """ Returns the initialized LLM model based on the service type."""
        if self.llm_service == "paid":
            # Initialize the Groq Chat model : Paid inference
            return ChatGroq(model=self.model_name, api_key=self.groq_api_key)
        else:
            # Ollama model initialization
            return Ollama(model=self.model_name)
