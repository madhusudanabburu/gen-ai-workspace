# Import the necessary libraries
import streamlit as st
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    AssemblyAIAudioTranscriptLoader
)
import assemblyai as aai
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import faiss

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "**"
os.environ["LANGCHAIN_PROJECT"] = "chat-with-documents"

model = os.environ.get("MODEL", "local_Meta_Llama_LLM_8B")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "mxbai-embed-large")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

chunk_size = 1024
chunk_overlap = 80
aai.settings.api_key = "**"

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".mp3": (AssemblyAIAudioTranscriptLoader, {})
}

embeddings_model_name = "mxbai-embed-large"
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embeddings_model_name)
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
db = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={},)

### Load Document ###
def load_document(file, my_bar):

    global db
    my_bar.progress(10, text="Creating the Vectorstore for uploaded documents")
    ext = "." + file.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file, **loader_args)

    my_bar.progress(10, text="Creating embeddings. May take some minutes...")
    if ext == '.mp3':
        config = aai.TranscriptionConfig(speaker_labels=True, auto_chapters=True, entity_detection=True)
        loader = AssemblyAIAudioTranscriptLoader(file_path=file, config=config)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(docs)
        for text in texts:
	        text.metadata = {"audio_url": text.metadata["audio_url"]}
    else:
        docs = loader.load()
        print("Num pages: ", len(docs))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(docs)

    db = FAISS.from_documents(texts, embedding=embeddings)
    my_bar.progress(100, text="Completed loading file " + file)
    return db

def create_chain():

    ### Setup the retriever tool ###

    global db

    retriever = db.as_retriever(search_type="mmr",
            search_kwargs={
                "k": 1
            })

    ### Setup the Model ###
    llm = Ollama(model="local_Meta_Llama_LLM_8B")

    ### Contextualize question ###

    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "use the following pieces of context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def side_bar():
    st.title("📝 Llama3 + Uploaded Document")
    file_uploaded = False
    progress_text = "File upload in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    global docs
    uploaded_file = st.file_uploader("'Upload a pdf/doc/html/text file' ", type=['pdf','doc','docx','html','txt', 'mp3', '.md'])
    if uploaded_file:
        my_bar.progress(1, text=progress_text)
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as file:
                file.write(uploaded_file.getvalue())
        print('file name : ', file.name)
        global db
        db = load_document(file.name, my_bar)
        file_uploaded = True
    else:
        st.write("No file was uploaded.")
    my_bar.empty()

    return file_uploaded

@st.experimental_fragment
def chat_window(file_uploaded):

    if file_uploaded:
        st.subheader("Chat with Documents", divider="red", anchor=False)

        #react_agent = create_agent()

        react_agent = create_chain()

        message_container = st.container(height=500, border=True)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if query := st.chat_input("Enter your query here..."):
            try:
                st.session_state.messages.append(
                    {"role": "user", "content": query})

                message_container.chat_message("user", avatar="😎").markdown(query)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner("model working..."):
                        response = react_agent.invoke({"input": query})
                        print(response)
                        #print(response["answer"])
                        message = response["answer"]
                        print("message -> ", message)

                        # stream response
                        st.write(message)
                        st.session_state.messages.append({"role": "assistant", "content": message})

            except Exception as e:
                #e.with_traceback()
                st.error(e, icon="⛔️")

def main():
    with st.sidebar:
        file_uploaded = side_bar()

    chat_window(file_uploaded)

if __name__ == "__main__":
    main()
