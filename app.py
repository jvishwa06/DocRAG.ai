import os
import tempfile
import streamlit as st
import requests
from dotenv import load_dotenv

from processors.documentprocessor import DocumentProcessor
from processors.ragprocessor import RAGProcessor

@st.cache_data(ttl=60)
def get_ollama_models(base_url="http://localhost:11434"):
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return sorted(model_names) if model_names else []
        return []
    except Exception:
        return []

@st.cache_data(ttl=300)
def get_groq_models(api_key):
    try:
        headers = {"Authorization": f"Bearer {api_key}","Content-Type": "application/json"}
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            text_models = []
            for model in models_data:
                model_id = model.get("id", "")
                if not any(exclude in model_id.lower() for exclude in ["whisper", "tts", "compound"]):
                    text_models.append(model_id)
            return sorted(text_models) if text_models else []
        return []
    except Exception:
        return []

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

for key, default_value in [
    ('rag_engine', None),
    ('current_provider', None), 
    ('current_model', None),
    ('processed_files', set())
]:
    if key not in st.session_state:
        st.session_state[key] = default_value

st.title("DocRAG.ai: Document Q/A Assistant")

with st.sidebar:
    st.title("LLM Configuration")
    llm_provider = st.selectbox(
        "Select LLM Provider",
        ["Groq", "Ollama"],
        help="Choose between cloud-based Groq or local Ollama"
    )
    
    if llm_provider == "Groq":
        api_key_input = st.text_input(
            "Groq API Key", 
            value=groq_api_key or "", 
            type="password",
            help="Enter your Groq API key or leave blank to use environment variable"
        )
        
        final_groq_api_key = api_key_input.strip() if api_key_input.strip() else groq_api_key
        
        if not final_groq_api_key:
            st.error("GROQ_API_KEY not found. Please enter your API key above or set it in your .env file.")
            st.stop()
        
        available_groq_models = get_groq_models(final_groq_api_key)
        
        if available_groq_models:
            default_index = 0
            if "llama-3.3-70b-versatile" in available_groq_models:
                default_index = available_groq_models.index("llama-3.3-70b-versatile")
            
            model_name = st.selectbox(
                "Groq Model",
                available_groq_models,
                index=default_index,
                help="Select from available Groq models"
            )
        else:
            st.warning("⚠️ Could not fetch Groq models. Using fallback list.")
            model_name = st.selectbox(
                "Groq Model (Fallback)",
                ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                index=0, 
                help="Fallback model list - check your API key if models don't load"
            )
    else:
        available_models = get_ollama_models()
        
        if available_models:
            model_name = st.selectbox(
                "Ollama Model",
                available_models,
                help="Select from locally available Ollama models"
            )
        else:
            st.warning("No Ollama models found locally or Ollama server not running")
            model_name = st.text_input(
                "Ollama Model", 
                value="llama3.2",
                help="Enter model name to pull automatically (e.g., llama3.2, mistral, codellama)"
            )
            st.info("If you enter a model name above, it will be automatically downloaded when you start using it.")

if llm_provider == "Groq":
    final_api_key = final_groq_api_key
else:
    final_api_key = None

@st.cache_resource
def get_document_processor():
    """Create and cache document processor"""
    return DocumentProcessor()

def get_rag_engine(llm_provider, model_name, api_key):
    """Create RAG engine with smart caching"""
    if (st.session_state.rag_engine is None or 
        st.session_state.current_provider != llm_provider or 
        st.session_state.current_model != model_name):
        
        try:
            doc_processor = get_document_processor()
            
            if llm_provider == "Groq":
                rag_engine = RAGProcessor(doc_processor,llm_provider="groq",model_name=model_name,api_key=api_key,temperature=0.0)
            else: 
                rag_engine = RAGProcessor(doc_processor,llm_provider="ollama",model_name=model_name,temperature=0.0)
            
            st.session_state.rag_engine = rag_engine
            st.session_state.current_provider = llm_provider
            st.session_state.current_model = model_name
            
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")
            if llm_provider == "Ollama":
                st.error("Make sure Ollama is running locally. You can start it with: `ollama serve`")
            st.stop()
    
    return st.session_state.rag_engine

try:
    rag_engine = get_rag_engine(llm_provider, model_name, final_api_key)
except Exception as e:
    st.error(f"Error getting RAG engine: {str(e)}")
    st.stop()

st.subheader("Upload Documents")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    current_files = {uploaded_file.name for uploaded_file in uploaded_files}
    new_files = current_files - st.session_state.processed_files
    
    if new_files:
        with st.spinner("Processing new documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name in new_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        doc_processor = get_document_processor()
                        result = doc_processor.process_document(tmp_path)
                        if result is None:
                            st.info(f"Document {uploaded_file.name} already processed - skipping")
                        else:
                            st.success(f"Successfully processed {uploaded_file.name}")
                        
                        st.session_state.processed_files.add(uploaded_file.name)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        os.unlink(tmp_path)

st.markdown("---")
st.subheader("Ask Questions")

question = st.text_input("Enter your question", placeholder="Type your question here...")
ask_button = st.button("Get Answer", type="primary")

if ask_button:
    if question:
        with st.spinner("Generating answer..."):
            try:
                answer = rag_engine.query(question)
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
    else:
        st.warning("Please enter a question first.")