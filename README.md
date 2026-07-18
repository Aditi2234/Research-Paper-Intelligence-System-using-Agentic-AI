# AI Research Paper Intelligence System

An AI-powered semantic search system for Machine Learning research papers that enables users to search, retrieve, summarize, and analyze research papers using transformer-based embeddings, FAISS vector search, and Large Language Models (LLMs).

---

## 🚀 Features

- 🔍 Semantic search using Sentence Transformers
- ⚡ Fast vector similarity search with FAISS
- 📄 Automatic paper summarization using Facebook BART
- 🏷️ Keyword extraction using KeyBERT
- 🤖 Named Entity Recognition (NER) using Hugging Face Transformers
- 💬 AI-powered research assistant with LangChain & Groq
- 📊 Research paper ranking based on semantic similarity
- 📚 Retrieval of relevant ML research papers from ArXiv

---

## 🏗️ Project Architecture

```text
                           User Query
                                │
                                ▼
                    Sentence Transformer
                  (all-MiniLM-L6-v2 Encoder)
                                │
                                ▼
                    Query Embedding (384-d)
                                │
                                ▼
                  FAISS Vector Similarity Search
                                │
          ┌─────────────────────┴─────────────────────┐
          │                                           │
          ▼                                           ▼
 Retrieve Top-K Papers                    Similarity Scores
          │
          ▼
Paper Title & Abstract
          │
          ├───────────────┬───────────────────┐
          ▼               ▼                   ▼
   BART Summarizer     KeyBERT          HuggingFace NER
          │               │                   │
          ▼               ▼                   ▼
    Paper Summary     Keywords          Named Entities
          │
          ▼
      LangChain Agent + Groq LLM
          │
          ▼
   Intelligent Research Assistant
```


## ⚙️ Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/AI_Research_Paper_Intelligence_System.git
cd AI_Research_Paper_Intelligence_System
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Launch the Streamlit application

```bash
streamlit run app.py
```

4. Open your browser and visit:

```
http://localhost:8501
```

---


## 📈 Future Enhancements
- PDF Upload & Analysis
- Citation Generation
- Research Recommendation System
- Multi-document Question Answering (RAG)
- Interactive Chatbot Interface
- Support for Multiple Research Domains

---

