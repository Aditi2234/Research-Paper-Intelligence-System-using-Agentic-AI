import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT
from langchain_groq import ChatGroq

st.set_page_config(page_title="Research Paper Assistant", layout="wide")
st.title("📚 Research Paper Search, Summarize & Compare Assistant")

@st.cache_resource
def load_data():
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    df = pd.DataFrame(dataset)
    df = df[['title', 'abstract']]
    df = df.head(15000)
    df["paper_text"] = df["title"] + " " + df["abstract"]
    df["paper_text"] = df["paper_text"].str.replace("\n", " ", regex=False).str.strip()
    return df

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_kw_model(_model):
    return KeyBERT(_model)

@st.cache_resource
def build_or_load_index(_df, _model):
    if os.path.exists("paper_embeddings.npy"):
        embeddings = np.load("paper_embeddings.npy")
    else:
        embeddings = _model.encode(
            _df["paper_text"].tolist(),
            batch_size=32,
            show_progress_bar=True
        )
        np.save("paper_embeddings.npy", embeddings)

    if os.path.exists("paper_faiss.index"):
        index = faiss.read_index("paper_faiss.index")
    else:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(384)
        index.add(embeddings)
        faiss.write_index(index, "paper_faiss.index")

    return embeddings, index

@st.cache_resource
def load_llm(api_key):
    return ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0)

def search_papers(query, model, index, k=5):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k)
    return D, I

def summarize_text(text, summarizer):
    summary = summarizer(text, max_length=120, min_length=40, do_sample=False)
    return summary[0]["summary_text"]

def extract_keywords(text, kw_model, top_n=5):
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=top_n
    )
    return keywords

def compare_papers(paper1_text, paper2_text, df, model, index, llm):
    embedding1 = model.encode([paper1_text])
    faiss.normalize_L2(embedding1)
    _, I1 = index.search(embedding1, 1)
    first_paper = df.iloc[I1[0][0]]

    embedding2 = model.encode([paper2_text])
    faiss.normalize_L2(embedding2)
    _, I2 = index.search(embedding2, 1)
    second_paper = df.iloc[I2[0][0]]

    comparison_prompt = f"""
Compare the following two research papers.

Paper 1
Title: {first_paper['title']}
Abstract: {first_paper['abstract']}

Paper 2
Title: {second_paper['title']}
Abstract: {second_paper['abstract']}

Compare them based on:
1. Research Objective
2. Methodology
3. Key Contributions
4. Advantages
5. Limitations
6. Applications

Present the comparison in a clear table.
"""
    response = llm.invoke(comparison_prompt)
    return first_paper, second_paper, response.content

st.sidebar.header("⚙️ Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
k_results = st.sidebar.slider("Number of results (k)", 1, 10, 5)

mode = st.sidebar.radio(
    "Choose a task",
    ["🔍 Search Papers", "📝 Search + Summarize", "🔑 Extract Keywords", "⚖️ Compare Papers", "🤖 Ask the Agent"]
)

with st.spinner("Loading dataset..."):
    df = load_data()

with st.spinner("Loading embedding model..."):
    embed_model = load_embedding_model()

with st.spinner("Building/loading FAISS index..."):
    embeddings, index = build_or_load_index(df, embed_model)

st.sidebar.success(f"Loaded {len(df)} papers ✅")

if mode == "🔍 Search Papers":
    st.subheader("🔍 Semantic Paper Search")
    query = st.text_input("Enter your search query", "deep learning for medical image analysis")

    if st.button("Search"):
        with st.spinner("Searching..."):
            D, I = search_papers(query, embed_model, index, k_results)

        for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            paper = df.iloc[idx]
            with st.expander(f"#{rank} — {paper['title']}  (score: {round(float(score),4)})"):
                st.write(paper["abstract"])

elif mode == "📝 Search + Summarize":
    st.subheader("📝 Search & Summarize Papers")
    query = st.text_input("Enter your search query", "Deep Learning in Medical Imaging")

    if st.button("Search & Summarize"):
        with st.spinner("Loading summarizer model..."):
            summarizer = load_summarizer()

        with st.spinner("Searching..."):
            D, I = search_papers(query, embed_model, index, k_results)

        for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            paper = df.iloc[idx]
            with st.expander(f"#{rank} — {paper['title']}  (score: {round(float(score),4)})"):
                st.markdown("**Abstract:**")
                st.write(paper["abstract"])
                with st.spinner(f"Summarizing paper {rank}..."):
                    summary = summarize_text(paper["abstract"], summarizer)
                st.markdown("**Summary:**")
                st.info(summary)


elif mode == "🔑 Extract Keywords":
    st.subheader("🔑 Keyword Extraction")
    text_input = st.text_area(
        "Paste text (or paper abstract) to extract keywords from",
        "Deep Learning for Medical Image Reconstruction"
    )
    top_n = st.slider("Number of keywords", 1, 15, 5)

    if st.button("Extract Keywords"):
        with st.spinner("Loading KeyBERT model..."):
            kw_model = load_kw_model(embed_model)

        with st.spinner("Extracting keywords..."):
            keywords = extract_keywords(text_input, kw_model, top_n)

        st.markdown("### Top Keywords")
        for rank, (keyword, score) in enumerate(keywords, start=1):
            st.write(f"**{rank}.** {keyword}  — relevance: `{round(score, 4)}`")


elif mode == "⚖️ Compare Papers":
    st.subheader("⚖️ Compare Two Papers")

    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to use this feature.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            paper1_query = st.text_input("Paper 1 (title/topic)", "Vision Transformer")
        with col2:
            paper2_query = st.text_input("Paper 2 (title/topic)", "Convolutional Neural Networks")

        if st.button("Compare"):
            with st.spinner("Loading LLM..."):
                llm = load_llm(groq_api_key)

            with st.spinner("Comparing papers..."):
                first_paper, second_paper, comparison = compare_papers(
                    paper1_query, paper2_query, df, embed_model, index, llm
                )

            st.markdown(f"**Paper 1 Matched:** {first_paper['title']}")
            st.markdown(f"**Paper 2 Matched:** {second_paper['title']}")
            st.markdown("### Comparison")
            st.markdown(comparison)


elif mode == "🤖 Ask the Agent":
    st.subheader("🤖 Ask the Research Agent")
    st.caption("Uses tools: search_and_summarize, extract_keywords (LangChain tool-calling)")

    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to use this feature.")
    else:
        from langchain_core.tools import tool
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

        with st.spinner("Loading models..."):
            summarizer = load_summarizer()
            kw_model = load_kw_model(embed_model)
            llm = load_llm(groq_api_key)

        @tool
        def search_and_summarize_tool(query: str, k: int = 5):
            """Search research papers from the FAISS database, retrieve top-k similar papers,
            summarize each using BART, and return structured results."""
            D, I = search_papers(query, embed_model, index, k)
            papers = []
            for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
                paper = df.iloc[idx]
                summary = summarize_text(paper["abstract"], summarizer)
                papers.append({
                    "rank": rank,
                    "similarity": round(float(score), 4),
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "summary": summary
                })
            return papers

        @tool
        def extract_keywords_tool(text: str, top_n: int = 5) -> str:
            """Extract the most important keywords from the given text using KeyBERT."""
            keywords = extract_keywords(text, kw_model, top_n)
            result = "Top Keywords:\n\n"
            for rank, (keyword, score) in enumerate(keywords, start=1):
                result += f"{rank}. {keyword} (Relevance Score: {round(score, 4)})\n"
            return result

        tools = [search_and_summarize_tool, extract_keywords_tool]
        llm_with_tools = llm.bind_tools(tools)

        user_query = st.text_input(
            "Ask something",
            "Find the top 3 research papers on Vision Transformer."
        )

        if st.button("Ask"):
            with st.spinner("Thinking..."):
                response = llm_with_tools.invoke(user_query)

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                st.caption(f"🔧 Calling tool: `{tool_name}` with args `{tool_args}`")

                if tool_name == "search_and_summarize_tool":
                    tool_result = search_and_summarize_tool.invoke(tool_args)
                    tool_result_str = str(tool_result)
                elif tool_name == "extract_keywords_tool":
                    tool_result_str = extract_keywords_tool.invoke(tool_args)
                else:
                    tool_result_str = "Unknown tool."

                tool_message = ToolMessage(
                    content=tool_result_str,
                    tool_call_id=tool_call["id"]
                )

                with st.spinner("Generating final answer..."):
                    final_response = llm.invoke([
                        SystemMessage(content="""
You are a helpful AI research assistant.
Rules:
1. Always use the tool output.
2. Never ignore tool results.
3. Present the complete tool output clearly.
4. Add a short explanation after the tool output if necessary.
"""),
                        HumanMessage(content=user_query),
                        response,
                        tool_message
                    ])

                st.markdown("### Answer")
                st.markdown(final_response.content)
            else:
                st.markdown("### Answer")
                st.markdown(response.content)
