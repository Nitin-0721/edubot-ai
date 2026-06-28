from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

def embed_docs(pdf_path, faiss_save_path: str):
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = DirectoryLoader(
        path=pdf_path, glob="*.pdf*", loader_cls=PyPDFLoader, use_multithreading=True
    )
    docs = loader.lazy_load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_documents([doc]))
    vectorstore = FAISS.from_documents(chunks, embedding=model)
    vectorstore.save_local(faiss_save_path)
