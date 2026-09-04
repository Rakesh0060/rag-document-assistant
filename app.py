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
    # Create folders if missing - this fixes your FileNotFoundError
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER, exist_ok=True)
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR, exist_ok=True)

    files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    
    if not files:
        return None

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
        except Exception as e:
            st.warning(f"Could not read {pdf_file}: {e}")

    if not all_texts:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(all_texts)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    return vectorstore

vectorstore = load_vectorstore()

if not vectorstore:
    st.warning(f"No PDFs found in {PDF_FOLDER}. Please upload PDFs to your GitHub repo in data/pdfs/ folder.")
    st.info("To fix: Go to GitHub > data/pdfs > Add file > Upload files > upload 1-2 PDFs")
else:
    st.success(f"Loaded {len(os.listdir(PDF_FOLDER))} PDFs!")
    
    groq_key = st.sidebar.text_input("Enter Groq API Key (free)", type="password")
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        if not groq_key:
            st.warning("Enter Groq API key in sidebar - get free from groq.com")
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])
            llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.3-70b-versatile")
            prompt = f"Answer based on context:\nContext: {context}\n\nQuestion: {query}\nAnswer:"
            response = llm.invoke(prompt)
            st.write("### Answer:")
            st.write(response.content)
