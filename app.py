import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from groq import Groq

st.set_page_config(page_title="RAG Assistant")
st.title("📄 RAG - 25 PDFs Supported")

# ---------- Embeddings (cached) ----------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------- Sidebar: upload ----------
st.sidebar.header("Upload (Max 10 at a time for free tier)")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if not uploaded_files:
    st.info("👉 Upload PDFs from sidebar. For 25 PDFs, upload 10+10+5 in 3 batches")
    st.stop()

if len(uploaded_files) > 10:
    st.sidebar.warning(f"You uploaded {len(uploaded_files)}. Processing first 10 only.")
    uploaded_files = uploaded_files[:10]

# ---------- Sidebar: API key ----------
groq_key = st.sidebar.text_input("Groq API Key", type="password")

# ---------- Fetch models available to this key ----------
available_models = []
if groq_key:
    try:
        client = Groq(api_key=groq_key.strip())
        available_models = sorted([m.id for m in client.models.list()])
        st.sidebar.caption(f"✅ {len(available_models)} models available")
    except Exception as e:
        st.sidebar.error(f"Key problem: {type(e).__name__}: {e}")

# ---------- Model selection ----------
default_model = "llama-3.3-70b-versatile"
options = available_models if available_models else [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
# put default first if it exists in the list
if default_model in options:
    options.remove(default_model)
    options.insert(0, default_model)

model_id = st.sidebar.selectbox("Select model", options)

# ---------- Process PDFs ----------
with st.spinner(f"Processing {len(uploaded_files)} PDFs... this takes ~60 sec"):
    all_texts = []
    for f in uploaded_files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        if text.strip():
            all_texts.append(text)

    if not all_texts:
        st.error("No extractable text found in the uploaded PDFs (are they scanned images?)")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents(all_texts)
    vectorstore = Chroma.from_documents(docs, get_embeddings())

st.success(f"✅ Loaded {len(uploaded_files)} PDFs! ({len(docs)} chunks)")

# ---------- Query ----------
query = st.text_input("Ask about your documents:")

if query and groq_key:
    try:
        with st.spinner("Retrieving & generating answer..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            context = "\n".join([d.page_content for d in retriever.invoke(query)])
            llm = ChatGroq(groq_api_key=groq_key.strip(), model=model_id)
            response = llm.invoke(f"Context: {context}\n\nQ: {query}\nAnswer briefly:")
        st.write(response.content)
    except Exception as e:
        st.error(f"Groq error: {type(e).__name__}: {e}")
elif query:
    st.warning("Enter Groq API key in sidebar")
