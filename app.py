import streamlit as st
import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="RAG Document Assistant", layout="wide")
st.title("📄 RAG Document Assistant")

PDF_FOLDER = "data/pdfs"
CHROMA_DIR = "chroma_db"

@st.cache_resource
def load_vectorstore():
    files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    all_texts = []
    for pdf_file in files:
        path = os.path.join(PDF_FOLDER, pdf_file)
        try:
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            if text.strip():
                all_texts.append(text)
        except:
            pass
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(all_texts)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    return vectorstore

vectorstore = load_vectorstore()

if vectorstore:
    st.success(f"Loaded {len(os.listdir(PDF_FOLDER))} PDFs!")
    groq_key = st.sidebar.text_input("Enter Groq API Key (free)", type="password")
    query = st.text_input("Ask a question about your documents:")
    if query:
        if not groq_key:
            st.warning("Enter Groq API key in sidebar")
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])
            llm = ChatGroq(groq_api_key=groq_key, model_name="openai/gpt-oss-20b")
            prompt = f"Answer based on context:\nContext: {context}\n\nQuestion: {query}\nAnswer:"
            response = llm.invoke(prompt)
            st.write("### Answer:")
            st.write(response.content)