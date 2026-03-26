# =====================================================
# RAG - Streamlit (FAISS + SentenceTransformers + HuggingFace Flan-T5)
# Corrige & Reformule correctement les reviews
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(layout="wide")
st.title("🤖 RAG - Review Reformulation (Deployable)")
st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_nlp.xlsx")
    df = df.dropna(subset=["avis_cor", "avis_en", "note"])
    df["avis_cor"] = df["avis_cor"].astype(str)\
        .str.replace(r'[\r\n]+', ' ', regex=True)\
        .str.replace(r'\s+', ' ', regex=True).str.strip()
    df["avis_en"] = df["avis_en"].astype(str)\
        .str.replace(r'[\r\n]+', ' ', regex=True)\
        .str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[(df["avis_cor"] != "") & (df["avis_en"] != "")]
    df["note"] = df["note"].astype(int)
    return df

df = load_data()
all_reviews_clean = df["avis_cor"].tolist()
all_reviews_en = df["avis_en"].tolist()
all_notes = df["note"].tolist()

review_db = [
    {"review_en": r_en, "note": note}
    for r_en, note in zip(all_reviews_en, all_notes)
]

st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# EMBEDDING MODEL
# =====================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# =====================================================
# EMBEDDINGS + FAISS
# =====================================================
@st.cache_data
def get_embeddings_batched(texts, batch_size=32, max_length=200):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_clean = [
            " ".join(str(text).split()[:max_length])
            for text in batch if str(text).strip()
        ]
        if not batch_clean:
            continue
        emb = model.encode(batch_clean)
        embeddings.extend(emb)
    return np.array(embeddings, dtype=np.float32)

@st.cache_resource
def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

with st.spinner("Generating embeddings and building FAISS index..."):
    all_embeddings = get_embeddings_batched(all_reviews_clean)
    if all_embeddings.size == 0:
        st.error("No embeddings generated.")
    else:
        index = build_faiss_index(all_embeddings)
        st.success(f"FAISS index ready with {index.ntotal} vectors.")

# =====================================================
# LOAD TEST DATA
# =====================================================
@st.cache_data
def load_test_reviews():
    df_test = pd.read_excel("data/reviews_clean.xlsx")
    df_test = df_test[df_test["type"] == "test"].reset_index(drop=True)
    return df_test

df_test = load_test_reviews()

# =====================================================
# RETRIEVAL
# =====================================================
def retrieve_similar_reviews(query, k=3):
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, k)
    return [review_db[i] for i in indices[0] if i < len(review_db)]

# =====================================================
# TEXT GENERATOR (Flan-T5)
# =====================================================
@st.cache_resource
def load_text_generator():
    return pipeline(
        task="text-generation",
        model="google/flan-t5-large",
        device_map="auto",
        max_new_tokens=200
    )

generator = load_text_generator()

# =====================================================
# PROMPT UTILS
# =====================================================
def correct_text(text):
    prompt = f"Correct spelling and grammar mistakes without changing the meaning:\n{text}"
    corrected = generator(prompt, max_new_tokens=200)[0]['generated_text']
    return corrected

def reformulate_text(text, similar_reviews):
    context = "\n".join([
        f"- Rating: {r['note']}★ Review: {r['review_en']}"
        for r in similar_reviews
    ]) if similar_reviews else "No similar reviews found."
    prompt = f"Reformulate this review to improve clarity and fluency, keeping the meaning. Consider these examples:\n{context}\n\nUser review: {text}"
    reformulated = generator(prompt, max_new_tokens=200)[0]['generated_text']
    return reformulated

# =====================================================
# USER INPUT
# =====================================================
st.header("🧪 Test a Review")
selected_review = st.selectbox("Select a review:", df_test["avis_en"].tolist())
user_input = st.text_area("Or write your own review:")
input_review = user_input.strip() if user_input.strip() else selected_review

# =====================================================
# GENERATE & DISPLAY RESULTS
# =====================================================
if st.button("Predict & Reformulate"):
    if input_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Processing..."):
            similar_reviews = retrieve_similar_reviews(input_review)
            corrected = correct_text(input_review)
            reformulated = reformulate_text(input_review, similar_reviews)

        st.success("✅ Done!")

        st.subheader("Original Review")
        st.write(input_review)

        st.subheader("Corrected Review")
        st.write(corrected)

        st.subheader("Reformulated Review")
        st.write(reformulated)

        st.subheader("🔍 Similar Reviews")
        for r in similar_reviews:
            st.markdown(f"- **{r['note']}★** | {r['review_en']}")

st.markdown("---")
st.caption("RAG app - deployable version (FAISS + SentenceTransformers + HuggingFace Flan-T5)")
