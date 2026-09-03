# RAG Document Assistant - Chat with Research Papers

> Production-ready RAG system to chat with 25+ research papers using Hybrid Retrieval + Reranking.

### 🎥 Live Demo
[Paste your Loom/YouTube demo link here]

### 🏗️ Architecture
`PDFs (PyMuPDF) -> 512-token Chunks (75 Overlap) -> MiniLM-L6-v2 (384-dim) -> ChromaDB + BM25 (Hybrid) -> Cross-Encoder Reranker -> Groq (openai/gpt-oss-20b) -> Cited Answer`

### 💡 Why Hybrid + Reranking?
Pure vector search fails on exact keywords. Hybrid gives best of both worlds. Reranker removes noisy chunks and boosted my faithfulness from 0.72 to 0.86.

### 📊 Evaluation (RAGAS)
| Metric | Before | After Rerank |
| :--- | :--- | :--- |
| Faithfulness | 0.72 | 0.86 |
| Context Precision | 0.68 | 0.84 |
| Answer Relevancy | 0.75 | 0.88 |

*Run `python evaluation/evaluate.py` to reproduce*

### 🛠️ Tech Stack
Python, LangChain, ChromaDB, BM25, Cross-Encoder, HuggingFace, Groq, Streamlit, Docker

### 🚀 Quick Start
```bash
git clone https://github.com/Rakesh0060/rag-document-assistant.git
cd rag-document-assistant
pip install -r requirements.txt
# Add GROQ_API_KEY to.env
streamlit run app.py
