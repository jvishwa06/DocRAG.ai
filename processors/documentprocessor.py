import os
from typing import List
import datetime
import uuid
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class DocumentProcessor:
    """loading, processing and embedding documents"""
    
    def __init__(self,embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",collection_name: str = "docrag.ai-collection",chunk_size: int = 1000,chunk_overlap: int = 200,):

        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        
        self.client = QdrantClient(location=":memory:")
        self.collection_name = collection_name
        self.vector_dimension = 384
        
        self.create_collection_if_not_exists()
        
        self.vector_store = QdrantVectorStore(client=self.client,collection_name=self.collection_name,embedding=self.embedding_model,)
    
    def create_collection_if_not_exists(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(collection_name=self.collection_name,vectors_config=VectorParams(size=self.vector_dimension, distance=Distance.COSINE))
        except Exception as e:
            self.client.create_collection(collection_name=self.collection_name,vectors_config=VectorParams(size=self.vector_dimension, distance=Distance.COSINE))
    
    def load_document(self, file_path: str) -> List[Document]:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        return loader.load()
    
    def process_document(self, file_path: str) -> List[Document]:
        document = self.load_document(file_path)
        
        splits = self.text_splitter.split_documents(document)
        
        doc_id = str(uuid.uuid4())
        
        filename = os.path.basename(file_path)
        
        for split in splits:
            if not split.metadata:
                split.metadata = {}
            
            split.metadata["document_id"] = doc_id
            split.metadata["filename"] = filename
            split.metadata["upload_date"] = datetime.datetime.now().isoformat()
        
        self.vector_store.add_documents(splits)
        
        return splits
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)
