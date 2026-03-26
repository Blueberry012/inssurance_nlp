# =====================================================
# RAG (Retrieval-Augmented Generation) - Streamlit
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import ollama
from tqdm import tqdm

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
# FAISS Index
# =====================================================
@st.cache_data
def get_embeddings_batched(texts, model="all-minilm", batch_size=16, max_length=200):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i+batch_size]
        batch_clean = []
        for text in batch:
            if not isinstance(text, str): continue
            text = " ".join(text.split()[:max_length])
            if text.strip(): batch_clean.append(text)
        if not batch_clean: continue
        response = ollama.embed(model=model, input=batch_clean)
        embeddings.extend(response["embeddings"])
    return np.array(embeddings, dtype=np.float32)

@st.cache_data
def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

with st.spinner("Generating embeddings and building FAISS index..."):
    all_embeddings = get_embeddings_batched(all_reviews_clean)
    if all_embeddings.size == 0:
        st.error("No embeddings generated. Check Ollama connection.")
    else:
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
def retrieve_similar_reviews(query, index, review_db, k=3):
    query_emb = get_embeddings_batched([query], batch_size=1)
    if query_emb.size == 0: return []
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, k)
    return [review_db[i] for i in indices[0] if i < len(review_db)]

# =====================================================
# Reformulate / RAG Prediction
# =====================================================
def reformulate_review(user_review, index, review_db, k=3):
    similar_reviews = retrieve_similar_reviews(user_review, index, review_db, k)
    context = "\n".join([
        f"- Rating: {r['note']}★\n  Review: {r['review_en']}"
        for r in similar_reviews
    ]) if similar_reviews else "No similar reviews found."
    
    prompt = f"""
You are an expert in customer feedback writing.

Your task:
1. Correct spelling and grammar mistakes
2. Improve clarity and fluency
3. Keep the original meaning
4. Do NOT invent new information
5. Keep it natural and human

### Similar existing reviews:
{context}

### User review:
{user_review}

### Output format:
Corrected version:
...

Reformulated version:
...
"""
    response = ollama.chat(
        model="mistral",
        messages=[{"role":"user","content":prompt}],
        options={"temperature":0.3}
    )
    return {
        "original": user_review,
        "response": response["message"]["content"],
        "similar_reviews": similar_reviews
    }

# =====================================================
# User Input / Test Review Selection (avec prompt éditable)
# =====================================================
st.header("🧪 Test a Single Review")

selected_review = st.selectbox(
    "Select a review from the test dataset:",
    df_test["avis_en"].tolist()
)

user_input = st.text_area("Or enter your own review here:", "")
input_review = user_input.strip() if user_input.strip() != "" else selected_review

# Génération du prompt par défaut
def generate_prompt(user_review, similar_reviews):
    context = "\n".join([
        f"- Rating: {r['note']}★\n  Review: {r['review_en']}"
        for r in similar_reviews
    ]) if similar_reviews else "No similar reviews found."

    prompt = f"""
You are an expert in customer feedback writing.

Your task:
1. Correct spelling and grammar mistakes
2. Improve clarity and fluency
3. Keep the original meaning
4. Do NOT invent new information
5. Keep it natural and human

### Similar existing reviews:
{context}

### User review:
{user_review}

### Output format:
Corrected version:
...

Reformulated version:
...
"""
    return prompt

# Bouton pour générer le prompt
if st.button("Generate Prompt Preview"):
    with st.spinner("Generating default prompt..."):
        similar_reviews = retrieve_similar_reviews(input_review, index, review_db, k=3)
        default_prompt = generate_prompt(input_review, similar_reviews)
    st.success("Prompt generated!")

    # Expander pour montrer et modifier le prompt
    edited_prompt = st.expander("Show / Edit Prompt")
    edited_prompt_text = st.text_area(
        "Edit the prompt if you have specific requests:",
        value=default_prompt,
        height=400
    )

# =====================================================
# Reformulate Review
# =====================================================
if st.button("Predict & Reformulate Review"):
    if input_review.strip() == "":
        st.warning("Please enter a review or select one from the test dataset.")
    else:
        # Utiliser le prompt édité si disponible, sinon le prompt par défaut
        prompt_to_use = edited_prompt_text if 'edited_prompt_text' in locals() else generate_prompt(input_review, retrieve_similar_reviews(input_review, index, review_db, k=3))

        with st.spinner("Retrieving similar reviews and reformulating..."):
            response = ollama.chat(
                model="mistral",
                messages=[{"role":"user","content":prompt_to_use}],
                options={"temperature":0.3}
            )
            similar_reviews_final = retrieve_similar_reviews(input_review, index, review_db, k=3)

        st.success("✅ Review processed!")

        st.subheader("Original Review")
        st.write(input_review)

        st.subheader("Reformulated Review")
        st.write(response["message"]["content"])

        if similar_reviews_final:
            st.subheader("🔍 Similar Reviews Retrieved")
            for r in similar_reviews_final:
                st.markdown(f"- **{r['note']}★** | Review: {r['review_en']}")

st.markdown("---")
st.caption("RAG - Retrieval-Augmented Generation for review reformulation and rating context")