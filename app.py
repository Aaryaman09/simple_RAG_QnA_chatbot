from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from icecream import ic

from src import get_config, track_queries_on_langsmith
from src.llm import FetchLLM
from src.chathistory import ChatHistoryManager
from src.prompt import PromptTemplate
from src.chains import DocumentChain, RetrievalChain
from src.chromaDB import VectorChromaDB
from src.data_ingestion import WebDataIngestion
from src.HGFembedding import HGFembedding

import os

class RAGApp:
    """
    A simple chatbot that uses Ollama or Groq.
    """
    def __init__(self, pass_keys):
        self.pass_keys = pass_keys
        # Initialize tracking on LangSmith 
        track_queries_on_langsmith(flag=pass_keys.get("track_queries_on_langsmith", False))

    def _get_path(self):
        if self.pass_keys.get("data_source_type") == "local_file":
            # Create a directory for local files if it doesn't exist
            path_components = self.pass_keys.get("source_url").get("local_file")
            return os.path.join(os.getcwd(),path_components.get("directory_name"), path_components.get("file_name"))
        elif self.pass_keys.get("data_source_type") == "remote_website_link":
            return self.pass_keys.get("source_url").get("remote_website_link")

    def fetch_input_data(self):
        # Fetch input data from the web using the WebDataIngestion class.
        web_data_ingestion = WebDataIngestion(
            url=self._get_path() 
        )
        documents = web_data_ingestion.fetch_data()

        return web_data_ingestion.split_documents(
            documents=documents,
            chunk_size=self.pass_keys.get("document_spliting_config").get("chunk_size"),
            chunk_overlap=self.pass_keys.get("document_spliting_config").get("chunk_overlap")
        )

    def fetch_prompt(self):
        prompt_obj = PromptTemplate()
        # return prompt_obj.get_prompt()      # This prompt forgets the chat history and just uses the input message
        return prompt_obj.get_chat_history_prompt()  # This prompt uses the chat history to contextualize the question

    def fetch_embeddings(self):
        """
        Fetch embeddings from the HuggingFace model specified in the pass_keys.
        """
        HGFembedding_obj = HGFembedding(
            model_name=self.pass_keys.get("HGFembedding").get("model_name")
        )
        return HGFembedding_obj.get_embeddings()
    
    def fetch_chroma_db(self, split_documents):
        """
        Fetch the Chroma vector database with the split documents and embeddings.
        """
        chroma_db_obj = VectorChromaDB(
            persist_directory=self.pass_keys.get("chroma_db_config").get("persist_directory"),
            documents=split_documents,
            embeddings=self.fetch_embeddings()
        )
        # return chroma_db_obj.get_retriever()    # This retriever forgets the chat history and just uses the input message
        return chroma_db_obj.get_history_aware_retriever(llm=self.fetch_llm())  # This LLM uses the chat history to contextualize the question (the prompt used in this will have chat history, LLM understand what we were talking about and will create a final prompt for the retriever to find the relevant context documents from the vector database). That is why llm is passed here.
                                                                                # Suppose question 1 : who is Elon musk?
                                                                                # And your question 2 : "what is his net worth?", llm will look at the chat history and understand that we were talking about Elon Musk, so it will create a prompt like "what is the net worth of Elon Musk?" and then final prompt will be sent to the retriever to find the relevant context documents from the vector database.
                                                                                # After receiving the context documents, the LLM will answer the question based on the context documents.
                                                                                # If you use line 70 along with line 49, LLM will not look at the chat history and just use the input message, it will not connect his = elon musk and give weird answers like "I don't know" or "I don't have any information about this person" 

    def fetch_llm(self):
        """
        Fetch the LLM model based on the service and model name specified in the pass_keys.
        """
        llm_obj = FetchLLM(
            llm_service=self.pass_keys.get("llm_service", "free"),
            model_name=self.pass_keys.get("model_dict").get(self.pass_keys.get("llm_service", "free")),
            groq_api_key=self.pass_keys.get("GROQ_API_KEY", "")
        )
        return llm_obj.get_model()
    
    def fetch_document_chain(self, llm, prompt):
        """
        Fetch a document chain using the provided LLM and prompt.
        """
        document_chain_obj = DocumentChain(llm=llm, prompt=prompt)
        return document_chain_obj.create_document_chain()

    def fetch_retriever_chain(self, chroma_db_retriever, document_chain):
        retrieval_chain_obj = RetrievalChain(
            retriever=chroma_db_retriever,
            document_chain=document_chain
        )

        return retrieval_chain_obj.create_retrieval_chain()

    def fetch_rag_chain(self):
        """
        Fetch a runnable that can handle message history for the chatbot.
        """

        # Fetching the input data from the web and splitting it into documents
        split_documents = self.fetch_input_data()

        # Fetching the prompt template
        prompt = self.fetch_prompt()            

        # Initialize the Chroma vector database with the split documents and embeddings
        chroma_db_retriever = self.fetch_chroma_db(split_documents)

        # Fetching the LLM
        llm = self.fetch_llm()

        # Fetching the document chain using the LLM and prompt
        document_chain = self.fetch_document_chain(llm=llm, prompt=prompt)

        # Create a retrieval chain using the Chroma retriever and document chain
        return self.fetch_retriever_chain(
            chroma_db_retriever=chroma_db_retriever,
            document_chain=document_chain
        )
    
        # Initialize the chat history manager and prompt template
        # history_manager = ChatHistoryManager()

        # # Create a RunnableWithMessageHistory to handle message history
        # with_message_history = RunnableWithMessageHistory(
        #     prompt_obj.get_prompt() | llm_obj.get_model(),              # Chain: prompt to model
        #     history_manager.get_session_history, 
        #     input_messages_key="messages")

        # # return the runnable and model name
        # return with_message_history


if __name__ == "__main__":

    pass_keys = get_config("key.json")

    chatbot = RAGApp(pass_keys=pass_keys)
    rag_chain = chatbot.fetch_rag_chain()

    chat_history = []

    print(f"Welcome to the ChatBot! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        key_config = get_config("key.json")
        # configuration = {
        #     "configurable": {
        #         "session_id": key_config.get("chat_session_id"),
        #     }
        # }

        # # Invoke the LLM with the human message
        # response = with_message_history.invoke(
        #     {
        #         "messages": [HumanMessage(content=user_input)],
        #         "language": user_input_language                      # key_config.get("language", "English")
        #     },
        #     config=configuration
        # )

        response = rag_chain.invoke(
            {
                "input": user_input,
                "chat_history": chat_history,
                "language": key_config.get('language', 'English')  # Default to English if not specified
                # "language": key_config.get('language')
            }
        )   

        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response['answer'])
        ])

        ic(response['answer'])

