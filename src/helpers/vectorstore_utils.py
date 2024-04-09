from langchain_community.vectorstores import Chroma
class VectorStore:
    def __init__(self, vector_store: str, index_name: str):
        self.vector_store = vector_store
        self.index_name = "/content/" + index_name

    def store_embeddings(self, embedding_function: str, docs: list):
        if self.vector_store == 'chroma':
            try:
                vector_index = Chroma.from_documents(documents = docs, embedding = embedding_function,  persist_directory = self.index_name)
            except:
                vector_index = Chroma.from_texts(texts = docs, embedding = embedding_function,  persist_directory = self.index_name)

            return vector_index.persist()
