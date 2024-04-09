
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
class Embeddings:
    def __init__(self, embedding_provider: str):
        self.embedding_provider = embedding_provider

    def get_embedding_function(self):
        if self.embedding_provider == 'huggingface':
            model_name =  "BAAI/bge-small-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            hf = HuggingFaceBgeEmbeddings(
                model_name = model_name,
                model_kwargs = model_kwargs,
                encode_kwargs = encode_kwargs,
            )
            return hf