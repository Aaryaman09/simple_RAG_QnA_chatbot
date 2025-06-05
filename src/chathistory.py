from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class ChatHistoryManager:
    """
    A simple chatbot that uses Ollama or Groq.
    """
    def __init__(self):
        self.store = {}

    def get_session_history(self, session_id) -> BaseChatMessageHistory:
        """
        Get the session chat message history.
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def get_runnable_with_history(self, rag_chain) -> RunnableWithMessageHistory:
        """
        Get a runnable with message history for the session.
        """

        return RunnableWithMessageHistory(
            rag_chain,   
            self.get_session_history, 
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer")
