import streamlit as st
import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="RAG Document Assistant", layout="wide")
st.title("📄 RAG Document Assistant - 25 PDFs Supported")

PDF_FOLDER = "data/pdfs"
CHROMA_DIR = "chroma_db"
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload up to 25 PDFs", type="pdf", accept_multiple_files=True)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdfs(pdf_files):
    all_texts = []
    for pdf_file in pdf_files:
        try:
            if hasattr(pdf_file, 'read'):
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            else:
                doc = fitz.open(pdf_file)
            text = ""
            for page in doc:
                text += page.get_text()
            if text.strip():
                all_texts.append(text)
        except Exception as e:
            st.warning(f"Skipped: {e}")
    if not all_texts:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(all_texts)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    return vectorstore

vectorstore = None
if uploaded_files:
    if len(uploaded_files) > 25:
        st.sidebar.error("Max 25 only!")
    else:
        with st.spinner(f"Processing {len(uploaded_files)} PDFs..."):
            vectorstore = process_pdfs(uploaded_files)
        st.sidebar.success(f"Loaded {len(uploaded_files)} PDFs!")
else:
    files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if files:
        with st.spinner(f"Loading {len(files)} PDFs..."):
            vectorstore = process_pdfs(files)

if not vectorstore:
    st.info("👈 Upload PDFs from sidebar to start")
else:
    groq_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    query = st.text_input("Ask a question about your documents:")
    if query:
        if not groq_key:
            st.warning("Enter Groq API key")
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])
            # FIXED MODEL NAME HERE
            llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant")
            prompt = f"Answer based on context:\nContext: {context}\n\nQuestion: {query}\nAnswer concisely:"
            response = llm.invoke(prompt)
            st.write("### Answer:")
            st.write(response.content)
