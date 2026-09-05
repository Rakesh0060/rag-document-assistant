import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="RAG - 25 PDFs Supported", layout="wide")
st.title("📄 RAG Document Assistant - 25 PDFs Supported")

# Sidebar
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Upload up to 25 PDFs", type="pdf", accept_multiple_files=True)
    
    st.divider()
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    
    model_name = st.selectbox(
        "Select model",
       ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"],
        index=0
    )

    # SIMPLE KEY CHECK - NO API CALL (fixes tuple error)
    if groq_key:
        if not groq_key.startswith("gsk_"):
            st.error("Invalid Key - must start with gsk_")
        else:
            st.success("Key format OK")

# Main
if not uploaded_files:
    st.info("👈 Upload PDFs from sidebar to start")
    st.stop()

# Process PDFs
with st.spinner(f"Processing {len(uploaded_files)} PDFs..."):
    full_text = ""
    for pdf_file in uploaded_files[:25]:
        try:
            pdf_bytes = pdf_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                full_text += page.get_text() + "\n"
        except Exception as e:
            st.warning(f"Error reading {pdf_file.name}: {e}")

    if not full_text.strip():
        st.error("No text found in PDFs")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(texts=chunks, embedding=embeddings)

st.success(f"✅ Loaded {len(uploaded_files)} PDFs - {len(chunks)} chunks created")

# Chat
query = st.text_input("Ask about your documents:", placeholder="What is CodePoisonRAG?")

if query:
    if not groq_key:
        st.warning("Please enter Groq API Key in sidebar")
    else:
        with st.spinner("Thinking..."):
            retriever = vectordb.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])
            
            llm = ChatGroq(groq_api_key=groq_key, model_name=model_name)
            prompt = f"""Use the following context to answer the question briefly.

Context:
{context}

Question: {query}
Answer:"""
            
            try:
                response = llm.invoke(prompt)
                st.markdown("### Answer")
                st.write(response.content)
                
                with st.expander("Sources"):
                    for i, d in enumerate(docs):
                        st.write(f"**Chunk {i+1}:** {d.page_content[:300]}...")
            except Exception as e:
                st.error(f"Groq Error: {e}")
                st.info("If 404 error -> check model name. If auth error -> create new key at console.groq.com/keys")
