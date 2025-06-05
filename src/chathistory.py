from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

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