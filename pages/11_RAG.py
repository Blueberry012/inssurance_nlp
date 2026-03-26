# =====================================================
# RAG (Retrieval-Augmented Generation) - Streamlit
# VERSION DEPLOYABLE (NO OLLAMA)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(layout="wide")
st.title("🤖 RAG - Review Reformulation (Deployable)")
st.markdown("---")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

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
# PROMPT
# =====================================================
def generate_prompt(user_review, similar_reviews):
    context = "\n".join([
        f"- Rating: {r['note']}★\n  Review: {r['review_en']}"
        for r in similar_reviews
    ]) if similar_reviews else "No similar reviews found."

    return f"""
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

# =====================================================
# USER INPUT
# =====================================================
st.header("🧪 Test a Review")

selected_review = st.selectbox(
    "Select a review:",
    df_test["avis_en"].tolist()
)

user_input = st.text_area("Or write your own review:")

input_review = user_input.strip() if user_input.strip() else selected_review

# =====================================================
# GENERATE PROMPT PREVIEW
# =====================================================
if st.button("Generate Prompt Preview"):
    similar = retrieve_similar_reviews(input_review)
    default_prompt = generate_prompt(input_review, similar)

    st.text_area("Editable Prompt", value=default_prompt, height=400)

# =====================================================
# PREDICTION
# =====================================================
if st.button("Predict & Reformulate"):

    if input_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Processing..."):

            similar_reviews = retrieve_similar_reviews(input_review)
            prompt = generate_prompt(input_review, similar_reviews)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            result = response.choices[0].message.content

        st.success("✅ Done!")

        st.subheader("Original Review")
        st.write(input_review)

        st.subheader("Reformulated Review")
        st.write(result)

        st.subheader("🔍 Similar Reviews")
        for r in similar_reviews:
            st.markdown(f"- **{r['note']}★** | {r['review_en']}")

st.markdown("---")
st.caption("RAG app - deployable version (FAISS + SentenceTransformers + OpenAI)")
