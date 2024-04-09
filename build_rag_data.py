import os
import argparse
import pandas as pd
from src.helpers.config import Config
from src.helpers.utils import Utils
from src.helpers.vectorstore_utils import VectorStore
from src.helpers.embedding_utils import Embeddings
from src.services.retrieval import Retrieval

parser = argparse.ArgumentParser(description="Build rag data by embedding and retrieving")
parser.add_argument('--vector_store', type=str, default='chroma', help='vector store')
parser.add_argument('--index_name', type=str, default='chroma_bge', help='index name')
parser.add_argument('--embedding_provider', type=str, default='huggingface', help='embedding provider')
parser.add_argument('--data_type', type = str, help='either train or test' )
parser.add_argument('--environment', type = str, choices=['local', 'colab'])
args = parser.parse_args()

def embed_and_build_rag_data():
    if args.data_type == "train":
        df = pd.read_csv(Config.train_filepath)
        if args.environment == "local":
            output_path = "src/output/train_with_rag.csv"
        else:
            output_path = "/content/train_with_rag.csv"
    else:
        df = pd.read_csv(Config.test_filepath)
        if args.environment == "local":
            output_path = "src/output/test_with_rag.csv"
        else:
            output_path = "/content/test_with_rag.csv"

    utils = Utils()
    vector_store = VectorStore(vector_store=Config.vector_store, index_name=Config.index_name)
    embedding_function = Embeddings(embedding_provider=Config.embedding_provider).get_embedding_function()
    retrieval = Retrieval(vector_store=Config.vector_store, index_name=Config.index_name)

    ####################################################[EMBEDDING]##########################################################################

    if not os.path.exists(f"index/{args.index_name}"):
        docs_all = utils.prepare_docs_list(folder_path=Config.folder_path)
        print(len(docs_all))
        vector_store.store_embeddings(embedding_function=embedding_function, docs=docs_all)
    else:
        print(f"Index {args.index_name} already exists")

    #########################################################[Retrieval]#####################################################################
    for index, row in df.iterrows():
        try:
            answer = retrieval.retrieve_only(embedding_function=embedding_function, query=row['Question Text'])
            string_answer = '\n'.join([f"Context {i+1}: {doc.page_content}, Reference: {doc.metadata['source']}, Paragraph: {doc.metadata['paragraph']}" for i, doc in enumerate(answer)])
            df.at[index, 'retrived_context'] = string_answer
            df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Error: {e} in row: {row}")

if __name__ == "__main__":
    embed_and_build_rag_data()





        


   