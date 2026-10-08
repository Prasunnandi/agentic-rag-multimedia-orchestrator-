from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = None

    def create_collection(self, texts, metadatas):
        self.db = Chroma.from_texts(texts, self.embeddings, metadatas=metadatas)
        return self.db.as_retriever()
