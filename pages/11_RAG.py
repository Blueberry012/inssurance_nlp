# =====================================================
# RAG - Streamlit (HuggingFace)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ------------------------
# Config
# ------------------------
st.set_page_config(layout="wide")
st.title("🤖 RAG - Review Reformulation (HuggingFace)")
st.markdown("---")

HF_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]  # ton token

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
review_db = [{"review_en": r_en, "note": note} for r_en, note in zip(all_reviews_en, all_notes)]

st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Embedding Model + FAISS
# =====================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

@st.cache_data
def get_embeddings(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = embedding_model.encode(batch)
        embeddings.extend(emb)
    return np.array(embeddings, dtype=np.float32)

@st.cache_resource
def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

with st.spinner("Generating embeddings and building FAISS index..."):
    embeddings = get_embeddings(all_reviews_clean)
    index = build_faiss_index(embeddings)
    st.success(f"FAISS index ready with {index.ntotal} vectors.")

# =====================================================
# Load HuggingFace Text2Text Model
# =====================================================
@st.cache_resource
def load_hf_model():
    model_name = "google/flan-t5-large"  # modèle text2text adapté
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

generator = load_hf_model()

# =====================================================
# Retrieval
# =====================================================
def retrieve_similar_reviews(query, k=3):
    query_emb = embedding_model.encode([query])
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, k)
    return [review_db[i] for i in indices[0] if i < len(review_db)]

# =====================================================
# Prompt Generation
# =====================================================
def generate_prompt(user_review, similar_reviews):
    context = "\n".join([f"- Rating: {r['note']}★\n  Review: {r['review_en']}" for r in similar_reviews]) \
        if similar_reviews else "No similar reviews found."
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
# User Input
# =====================================================
st.header("🧪 Test a Review")
selected_review = st.selectbox("Select a review:", df["avis_en"].tolist())
user_input = st.text_area("Or write your own review:")
input_review = user_input.strip() if user_input.strip() else selected_review

if st.button("Generate Prompt Preview"):
    similar_reviews = retrieve_similar_reviews(input_review)
    prompt = generate_prompt(input_review, similar_reviews)
    st.text_area("Prompt (editable)", value=prompt, height=400)

# =====================================================
# Prediction
# =====================================================
if st.button("Predict & Reformulate"):
    if input_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        similar_reviews = retrieve_similar_reviews(input_review)
        prompt = generate_prompt(input_review, similar_reviews)
        with st.spinner("Generating corrected and reformulated review..."):
            output = generator(prompt, max_length=512, do_sample=True, temperature=0.3)[0]["generated_text"]

        st.subheader("Original Review")
        st.write(input_review)
        st.subheader("Reformulated Review")
        st.write(output)
        st.subheader("🔍 Similar Reviews")
        for r in similar_reviews:
            st.markdown(f"- **{r['note']}★** | {r['review_en']}")
