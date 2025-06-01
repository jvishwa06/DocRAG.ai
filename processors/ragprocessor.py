from typing import Optional
import requests
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from processors.documentprocessor import DocumentProcessor

class RAGProcessor:
    """Handles contextual retrieval and generation"""
    
    def __init__(self, 
                 document_processor: DocumentProcessor, 
                 llm_provider: str = "groq", 
                 model_name: str = "llama-3.3-70b-versatile", 
                 api_key: Optional[str] = None, 
                 temperature: float = 0.0, 
                 retrieval_k: int = 10, 
                 ollama_base_url: str = "http://localhost:11434"):

        self.document_processor = document_processor
        self.retrieval_k = retrieval_k
        self.llm_provider = llm_provider
        self.ollama_base_url = ollama_base_url
        
        if llm_provider.lower() == "ollama":
            self.ensure_ollama_model(model_name)
            self.llm = ChatOllama(model=model_name,base_url=ollama_base_url,temperature=temperature)
        else:
            self.llm = ChatGroq(model_name=model_name,api_key=api_key,temperature=temperature)
        
        self.prompt = ChatPromptTemplate.from_template("""
                    You are a helpful and accurate document assistant. Answer the user's question
                    based ONLY on the provided context. If the context doesn't contain the answer,
                    say "I don't have enough information to answer this question." Don't make up information.

                    Context:
                    {context}

                    User Question: {question}

                    Helpful Answer:
                    """)
        
        self.rag_chain = (
            {"context": lambda x: self.get_context(x["question"]), "question": lambda x: x["question"]} 
            | self.prompt | self.llm | StrOutputParser()
            )
    
    def ensure_ollama_model(self, model_name: str) -> bool:
        """Check if Ollama model exists and pull it if not available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                model_exists = any(model_name in [name, name.split(':')[0], f"{model_name}:latest"] for name in model_names)
                
                if model_exists:
                    return True
            
            print(f"Model '{model_name}' not found locally. Attempting to pull...")
            response = requests.post(f"{self.ollama_base_url}/api/pull",json={"name": model_name},stream=True,timeout=300)
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if data.get("status") == "success":
                                print(f"Successfully pulled model '{model_name}'")
                                return True
                            elif "error" in data:
                                raise Exception(f"Error pulling model: {data['error']}")
                        except json.JSONDecodeError:
                            continue
            else:
                raise Exception(f"Failed to pull model. Status code: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Model '{model_name}' could not be ensured. Please check the model name or pull manually: ollama pull {model_name}")
        
        return False
    
    def get_context(self, query: str) -> str:
        docs = self.document_processor.similarity_search(query)
        return "\n\n".join([doc.page_content for doc in docs])
    
    def query(self, question: str) -> str:
        return self.rag_chain.invoke({"question": question})