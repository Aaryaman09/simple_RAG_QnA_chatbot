from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class PromptTemplate:
    """
    A class to create a prompt template for a chatbot.
    This template includes a system message and a placeholder for user messages.
    """
    
    def __init__(self):
        self.system_message = "You are a helpful assistant. " \
        "Answer the user's questions to the best of your ability in language : {language}. " \
        "Answer only if you are sure about the answer and should only come from the context provided. " \
        "If you don't find any relevant information, say 'I don't know'. \n\n {context}"

    def get_prompt(self) -> ChatPromptTemplate:
        """
        Returns the prompt template with the specified language.
        """

        return ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", "{input}")
        ])

    def get_chat_history_prompt(self):

        return ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

    def get_contextualize_q_system_prompt(self):
        contextualize_q_system_prompt = "Given the following chat history, " \
            " and latest user question, " \
            "which might reference context in the chat history, " \
            "formulate a standalone question which can be understood without the chat history. " \
            "The question should be concise and clear, " \
            "DO NOT ANSWER THE QUESTION, " \
            "just reformulate it if needed and otherwise return it as is" \

        return ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])