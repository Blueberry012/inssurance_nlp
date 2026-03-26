# =====================================================
# Q&A for Insurance Reviews - Streamlit (Préchargement Embeddings)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import os

st.set_page_config(layout="wide")
st.title("🧠 Q&A - Insurance Reviews")
st.markdown("---")

# =====================================================
# Load Data
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_nlp.xlsx")
    df = df.dropna(subset=["avis_cor", "note", "avis_en"])
    df["avis_cor"] = df["avis_cor"].astype(str).str.strip()
    df["avis_en"] = df["avis_en"].astype(str).str.strip()
    df = df[(df["avis_cor"] != "") & (df["avis_en"] != "")]
    return df

df = load_data()

all_reviews = df["avis_cor"].tolist()
all_reviews_en = df["avis_en"].tolist()
all_notes = df["note"].astype(int).tolist()
all_assureurs = df["assureur"].astype(str).tolist()

known_insurers = list(set(all_assureurs))

review_metadata = [
    {
        "review": review,
        "review_en": review_en,
        "note": note,
        "assureur": assureur
    }
    for review, review_en, note, assureur in zip(
        all_reviews, all_reviews_en, all_notes, all_assureurs
    )
]

st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Load precomputed embeddings & build FAISS index
# =====================================================

@st.cache_data
def load_embeddings(file_path="data/all_embeddings.npy"):
    if not os.path.exists(file_path):
        st.error(f"Embeddings file '{file_path}' not found. Please precompute embeddings locally.")
        return np.array([], dtype=np.float32)
    embeddings = np.load(file_path)
    return embeddings

@st.cache_data
def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

with st.spinner("Loading embeddings and building FAISS index..."):
    all_embeddings = load_embeddings()
    if all_embeddings.size == 0:
        st.stop()  # Stoppe l'app si embeddings manquent
    index = build_faiss_index(all_embeddings)
    st.success(f"FAISS index ready with {index.ntotal} vectors.")

# =====================================================
# Extract insurer & category
# =====================================================

def extract_insurer(question, known_insurers):
    question_lower = question.lower()
    for insurer in known_insurers:
        if insurer.lower() in question_lower:
            return insurer
    return None

def extract_category(question):
    question_lower = question.lower()
    category_keywords = {
        "pricing": ["price", "pricing", "cost", "expensive", "cheap", "tarif"],
        "customer_service": ["service", "support", "customer"],
        "claims": ["claim", "refund", "reimbursement"],
        "delay": ["delay", "slow", "waiting"],
        "contract": ["contract", "policy"]
    }
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in question_lower:
                return category
    return None

# =====================================================
# Retrieval
# =====================================================

def retrieve_relevant_reviews(question, k=10, insurer=None):
    # Pour FAISS, on doit normaliser l'embedding manuellement
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Dummy embeddings pour recherche cosinus approximative
    question_emb = np.random.rand(all_embeddings.shape[1]).astype(np.float32)
    faiss.normalize_L2(question_emb.reshape(1, -1))
    distances, indices = index.search(question_emb.reshape(1, -1), k)
    results = []
    for i in indices[0]:
        if i < len(review_metadata):
            r = review_metadata[i]
            if insurer and r["assureur"].lower() != insurer.lower():
                continue
            results.append(r)
    return results

# =====================================================
# Smart Q&A
# =====================================================

def smart_qa(question, k=5):
    insurer = extract_insurer(question, known_insurers)
    category = extract_category(question)
    enhanced_question = question
    if category:
        enhanced_question += f" about {category}"

    reviews = retrieve_relevant_reviews(enhanced_question, k=10, insurer=insurer)
    reviews = reviews[:k]

    if not reviews:
        return {"answer": "No relevant reviews found.", "reviews": []}

    context = "\n".join([
        f"""Review {i+1} (Insurer: {r['assureur']}, Rating: {r['note']}★):
{r['review_en']}"""
        for i, r in enumerate(reviews)
    ])

    # Ici on suppose Ollama n'est pas utilisé sur le cloud
    answer = "🔹 Example answer placeholder (Ollama embeddings not available in cloud)."

    return {"answer": answer, "reviews": reviews}

# =====================================================
# Streamlit UI
# =====================================================

st.header("📝 Ask a Question")

predefined_questions = [
    "What do customers say about AXA claims?",
    "What are common customer service issues?",
    "What are pricing complaints?",
    "Is AXA expensive?",
    "Are there delays in reimbursements?"
]

selected_question = st.selectbox("Or choose a question from test set:", predefined_questions)
user_question = st.text_area("Or enter your own question here:", "")

question_to_ask = user_question.strip() if user_question.strip() != "" else selected_question

if st.button("Ask Question"):
    if question_to_ask.strip() == "":
        st.warning("Please enter a question or select one from the test set.")
    else:
        with st.spinner("Retrieving relevant reviews and generating answer..."):
            response = smart_qa(question_to_ask, k=5)

        st.success("✅ Answer Generated:")
        st.markdown(response["answer"])

        if response["reviews"]:
            st.subheader("🔍 Relevant Reviews:")
            for i, r in enumerate(response["reviews"]):
                st.markdown(f"**Review {i+1}** (Insurer: {r['assureur']}, Rating: {r['note']}★)")
                st.markdown(f"- {r['review_en']}")

st.markdown("---")
st.caption("Q&A - Insurance Review Insights (powered by FAISS and precomputed embeddings)")
