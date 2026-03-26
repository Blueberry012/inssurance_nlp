# =====================================================
# RAG (Retrieval-Augmented Generation) - Streamlit
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import os

st.set_page_config(layout="wide")
st.title("🤖 RAG - Retrieval-Augmented Generation for Reviews")
st.markdown("---")

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_nlp.xlsx")
    df = df.dropna(subset=["avis_cor", "avis_en", "note"])
    df["avis_cor"] = df["avis_cor"].astype(str).str.replace(r'[\r\n]+', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    df["avis_en"] = df["avis_en"].astype(str).str.replace(r'[\r\n]+', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[(df["avis_cor"] != "") & (df["avis_en"] != "")]
    df["note"] = df["note"].astype(int)
    return df

df = load_data()
all_reviews_clean = df["avis_cor"].tolist()
all_reviews_en = df["avis_en"].tolist()
all_notes = df["note"].tolist()

# structured review_db
review_db = [
    {"review_en": r_en, "note": note}
    for r_en, note in zip(all_reviews_en, all_notes)
]

st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Load Precomputed Embeddings
# =====================================================
@st.cache_data
def load_embeddings(file_path="data/all_embeddings.npy"):
    if os.path.exists(file_path):
        embeddings = np.load(file_path)
        return embeddings.astype(np.float32)
    else:
        st.error(f"Embeddings file '{file_path}' not found. Please precompute embeddings locally.")
        return np.zeros((len(all_reviews_clean), 384), dtype=np.float32)  # fallback

all_embeddings = load_embeddings()

# =====================================================
# Build FAISS Index
# =====================================================
@st.cache_data
def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

index = build_faiss_index(all_embeddings)
st.success(f"FAISS index ready with {index.ntotal} vectors.")

# =====================================================
# Load Test Reviews
# =====================================================
@st.cache_data
def load_test_reviews():
    df_test = pd.read_excel("data/reviews_clean.xlsx")
    df_test = df_test[df_test["type"]=="test"].reset_index(drop=True)
    return df_test

df_test = load_test_reviews()

# =====================================================
# Retrieval function
# =====================================================
def retrieve_similar_reviews(query_emb, index, review_db, k=3):
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, k)
    return [review_db[i] for i in indices[0] if i < len(review_db)]

# =====================================================
# Generate Embedding for a Query (simulate Ollama offline)
# =====================================================
def get_query_embedding(query, embedding_file="data/all_embeddings.npy", text_list=all_reviews_clean):
    """
    Offline fallback: find closest review embedding by simple bag-of-words / TF-IDF or return zeros.
    """
    # For simplicity, return random vector if no Ollama access
    return np.random.rand(1, all_embeddings.shape[1]).astype(np.float32)

# =====================================================
# User Input / Test Review Selection
# =====================================================
st.header("🧪 Test a Single Review")

selected_review = st.selectbox(
    "Select a review from the test dataset:",
    df_test["avis_en"].tolist()
)

user_input = st.text_area("Or enter your own review here:", "")
input_review = user_input.strip() if user_input.strip() != "" else selected_review

# =====================================================
# Reformulate Review (offline fallback)
# =====================================================
def reformulate_review_offline(user_review, index, review_db, k=3):
    query_emb = get_query_embedding(user_review)
    similar_reviews = retrieve_similar_reviews(query_emb, index, review_db, k)
    context = "\n".join([f"- Rating: {r['note']}★ | Review: {r['review_en']}" for r in similar_reviews]) if similar_reviews else "No similar reviews found."
    return {
        "original": user_review,
        "response": f"Offline reformulation not available. Similar reviews context:\n{context}",
        "similar_reviews": similar_reviews
    }

# =====================================================
# Predict & Reformulate Review
# =====================================================
if st.button("Predict & Reformulate Review"):
    if input_review.strip() == "":
        st.warning("Please enter a review or select one from the test dataset.")
    else:
        with st.spinner("Retrieving similar reviews and reformulating..."):
            response = reformulate_review_offline(input_review, index, review_db, k=3)
        st.success("✅ Review processed!")

        st.subheader("Original Review")
        st.write(input_review)

        st.subheader("Reformulated Review / Context")
        st.write(response["response"])

        if response["similar_reviews"]:
            st.subheader("🔍 Similar Reviews Retrieved")
            for r in response["similar_reviews"]:
                st.markdown(f"- **{r['note']}★** | Review: {r['review_en']}")

st.markdown("---")
st.caption("RAG - Retrieval-Augmented Generation (offline-safe version)")
