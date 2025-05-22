from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from processors.documentprocessor import DocumentProcessor

class RAGProcessor:
    """Handles contextual retrieval and generation"""
    
    def __init__(self,document_processor: DocumentProcessor,model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None,temperature: float = 0.0, retrieval_k: int = 8):

        self.document_processor = document_processor
        self.retrieval_k = retrieval_k
        
        self.llm = ChatGroq(model_name=model_name,api_key=api_key,temperature=temperature,)
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful and accurate document assistant. Answer the user's question
        based ONLY on the provided context. If the context doesn't contain the answer,
        say "I don't have enough information to answer this question." Don't make up information.
        
        Context:
        {context}
        
        User Question: {question}
        
        Helpful Answer:
        """)
        
        self.rag_chain = ({"context": lambda input_dict: self.get_context(input_dict["question"]), "question": lambda input_dict: input_dict["question"]} | self.prompt | self.llm | StrOutputParser())
    
    def get_context(self, query: str) -> str:
        docs = self.document_processor.similarity_search(query)
        return "\n\n".join([doc.page_content for doc in docs])
    
    def query(self, question: str) -> str:
        return self.rag_chain.invoke({"question": question})