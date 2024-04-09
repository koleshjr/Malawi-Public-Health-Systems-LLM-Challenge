from langchain_community.vectorstores import Chroma
class Retrieval:
    def __init__(self, vector_store, index_name, top_k: int = 1):
        self.vector_store = vector_store
        self.index_name = "/content/" + index_name
        self.top_k = top_k

    def retrieve_only(self, query: str, embedding_function: str):
        if self.vector_store == 'chroma':
            vector_index = Chroma(persist_directory = self.index_name, embedding_function = embedding_function)

        docs = vector_index.similarity_search(query, k = 3)
        return docs

