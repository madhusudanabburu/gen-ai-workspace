from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os
import glob
import bs4
from constants import CHROMA_SETTINGS

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'nomic-embed-text')
chunk_size = 50
chunk_overlap = 10

### Index ###
def get_website_documents():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    loader = WebBaseLoader(urls, bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ))

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)

    return texts

def init_chromadb():
    print("Creating the Vectorstore for website documents")

    if not os.path.exists(persist_directory):
        os.mkdir(persist_directory)

    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embeddings_model_name)
    #embeddings = NomicEmbeddings(model=embeddings_model_name, inference_mode="local")
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(documents=get_website_documents(), embedding=embeddings, collection_name='website_docs', persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"}, client_settings=CHROMA_SETTINGS)
    client = chromadb.PersistentClient(path=persist_directory, settings=CHROMA_SETTINGS)
    collection = client.get_collection(name="website_docs") 
    print("Total objects in collection - website_docs ", collection.count())

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    list_index_files = glob.glob(os.path.join(persist_directory, '*/*.bin'))
    list_index_files += glob.glob(os.path.join(persist_directory, '*/*.pkl'))
    # At least 3 documents are needed in a working vectorstore
    if len(list_index_files) > 3:
        return True
    return False

def main():
    client = chromadb.PersistentClient(path=persist_directory, settings=CHROMA_SETTINGS)
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        collection = client.get_collection(name="website_docs")
        print(f"DB existing ...")
        print(collection.count())
    else:
        # Create and store locally vectorstore
        print("Creating new ChromaDB")
        init_chromadb()

    print(f"Ingestion complete! You can now run chat-langgraph.py to query your documents")


if __name__ == "__main__":
    main()
