import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="RAG - 25 PDFs Supported", layout="wide")
st.title("📄 RAG Document Assistant - 25 PDFs")

with st.sidebar:
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    groq_key = st.text_input("Groq API Key", type="password")
    model_name = st.selectbox("Select model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])

if uploaded:
    docs_text = ""
    for pdf in uploaded[:25]:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            docs_text += page.get_text()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(docs_text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embeddings)
    
    st.success(f"✅ Loaded {len(uploaded)} PDFs!")

    query = st.text_input("Ask about your documents:")
    if query and groq_key:
        if not groq_key.startswith("gsk_"):
            st.error("Invalid API Key - must start with gsk_")
        else:
            retriever = vectordb.as_retriever()
            context_docs = retriever.get_relevant_documents(query)
            context = "\n".join([d.page_content for d in context_docs])
            
            llm = ChatGroq(groq_api_key=groq_key, model_name=model_name)
            response = llm.invoke(f"Context: {context}\n\nQ: {query}\nAnswer briefly:")
            st.write(response.content)
else:
    st.info("Upload PDFs from sidebar to start")
