from langchain_core.prompts import ChatPromptTemplate

class PromptTemplate:
    """
    A class to create a prompt template for a chatbot.
    This template includes a system message and a placeholder for user messages.
    """
    
    def __init__(self):
        pass

    def get_prompt(self) -> ChatPromptTemplate:
        """
        Returns the prompt template with the specified language.
        """

        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's questions to the best of your ability in language : {language}. Answer only if you are sure about the answer. If you are not sure, say 'I don't know'. \n\n {context}"),
            ("human", "{input}"),  ## MessagesPlaceholder(variable_name="input") is an alternative way for user input but used for message history scenario.
        ])

