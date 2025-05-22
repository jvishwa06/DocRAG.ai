import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from processors.documentprocessor import DocumentProcessor
from processors.ragprocessor import RAGProcessor

try:
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please set it.")
        st.stop()
except Exception as e:
    st.error(f"Error loading environment variables: {str(e)}")
    st.stop()

try:
    doc_processor = DocumentProcessor()
    rag_engine = RAGProcessor(doc_processor,api_key=groq_api_key)
except Exception as e:
    st.error(f"Error initializing RAG system: {str(e)}")
    st.stop()

st.title("DocRAG.ai: Document Q/A Assistant")

st.subheader("Upload Documents")
try:
    uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True, on_change=None)

    if uploaded_files:
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        doc_processor.process_document(tmp_path)
                        st.success(f"Successfully processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception as e:
                            st.warning(f"Could not remove temporary file: {str(e)}")
                except Exception as e:
                    st.error(f"Error handling file {uploaded_file.name}: {str(e)}")
except Exception as e:
    st.error(f"Error in file upload functionality: {str(e)}")

st.markdown("---")
st.subheader("Ask Questions")

try:
    question = st.text_input("Enter your question and press Enter")

    if question:
        with st.spinner("Generating answer..."):
            try:
                answer = rag_engine.query(question)
                st.subheader("Answer")
                st.write(answer)
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
except Exception as e:
    st.error(f"Error in question answering functionality: {str(e)}")

with st.sidebar:
    st.title("About DocRAG.ai")
    st.markdown("""
    **DocRAG.ai** is a document Q/A RAG system that allows you to:
    
    - Upload PDF or DOCX documents
    - Ask questions about your documents
    - Get accurate answers
    
    **System Details:**
    - LLM: Llama-3.3-70b-versatile
    - Vector DB: Qdrant
    - Embedding Model: all-MiniLM-L6-v2
    """)