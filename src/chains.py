from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# create a class for DocumentChain that combines the functionality of creating a stuff documents chain
class DocumentChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def create_document_chain(self):
        """
        Create a document chain using the provided LLM and prompt.
        """
        return create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt
        )
    
class RetrievalChain:
    def __init__(self, retriever, document_chain):
        self.retriever = retriever
        self.document_chain = document_chain

    def create_retrieval_chain(self):
        """
        Create a retrieval chain using the provided retriever and document chain.
        """
        return create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=self.document_chain
        )