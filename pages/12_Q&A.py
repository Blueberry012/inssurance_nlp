# =====================================================
# Q&A for Insurance Reviews - Streamlit
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import ollama
from tqdm import tqdm

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
# Embeddings & FAISS Index
# =====================================================

@st.cache_data
def get_embeddings_batched(texts, model="all-minilm", batch_size=16, max_length=200):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        batch_clean = []
        for text in batch:
            if not isinstance(text, str): continue
            text = text.strip()
            if not text: continue
            truncated = " ".join(text.split()[:max_length])
            batch_clean.append(truncated)
        if not batch_clean: continue
        try:
            response = ollama.embed(model=model, input=batch_clean)
            embeddings.append(np.array(response["embeddings"], dtype=np.float32))
        except Exception as e:
            print(f"Error: {e}")
            continue
    if not embeddings: return np.array([], dtype=np.float32).reshape(0, 0)
    return np.vstack(embeddings)

@st.cache_data
def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

with st.spinner("Generating embeddings and building FAISS index..."):
    all_embeddings = get_embeddings_batched(all_reviews)
    if all_embeddings.size == 0:
        st.error("No embeddings generated. Check Ollama connection.")
    else:
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
    question_embedding = get_embeddings_batched([question], batch_size=1)
    if question_embedding.size == 0: return []
    faiss.normalize_L2(question_embedding)
    distances, indices = index.search(question_embedding, k)
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
Test: {r['review_en']}"""
        for i, r in enumerate(reviews)
    ])

    prompt = f"""
You are an expert insurance analyst.

### Reviews:
{context}

### Question:
{question}

### Instructions:
- Answer in 2-3 sentences
- Add bullet points if relevant
- Mention insurer(s)

### Answer:
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3}
    )

    return {"answer": response["message"]["content"], "reviews": reviews}

# =====================================================
# Streamlit UI with editable prompt
# =====================================================

st.header("📝 Ask a Question")

# Option: choose a predefined question
predefined_questions = [
    "What do customers say about AXA claims?",
    "What are common customer service issues?",
    "What are pricing complaints?",
    "Is AXA expensive?",
    "Are there delays in reimbursements?"
]

selected_question = st.selectbox("Or choose a question from test set:", predefined_questions)

# Option: enter custom question
user_question = st.text_area("Or enter your own question here:", "")

# Decide final question
question_to_ask = user_question.strip() if user_question.strip() != "" else selected_question

# Generate default prompt for preview / editing
def generate_qa_prompt(question, reviews):
    context = "\n".join([
        f"""Review {i+1} (Insurer: {r['assureur']}, Rating: {r['note']}★):
Test: {r['review_en']}"""
        for i, r in enumerate(reviews)
    ])
    prompt = f"""
You are an expert insurance analyst.

### Reviews:
{context}

### Question:
{question}

### Instructions:
- Answer in 2-3 sentences
- Add bullet points if relevant
- Mention insurer(s)

### Answer:
"""
    return prompt

# Bouton pour générer prompt preview
if st.button("Generate Prompt Preview"):
    with st.spinner("Generating default prompt..."):
        relevant_reviews = retrieve_relevant_reviews(question_to_ask, k=5)
        default_prompt = generate_qa_prompt(question_to_ask, relevant_reviews)
    st.success("Prompt generated!")

    # Expander pour montrer et éditer le prompt
    edited_prompt_expander = st.expander("Show / Edit Prompt")
    edited_prompt_text = st.text_area(
        "Edit the prompt if you have specific instructions:",
        value=default_prompt,
        height=400
    )

# =====================================================
# Ask Question with optional edited prompt
# =====================================================
if st.button("Ask Question"):
    if question_to_ask.strip() == "":
        st.warning("Please enter a question or select one from the test set.")
    else:
        # Use edited prompt if available, otherwise default
        prompt_to_use = edited_prompt_text if 'edited_prompt_text' in locals() else generate_qa_prompt(question_to_ask, retrieve_relevant_reviews(question_to_ask, k=5))

        with st.spinner("Retrieving relevant reviews and generating answer..."):
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt_to_use}],
                options={"temperature": 0.3}
            )
            reviews_final = retrieve_relevant_reviews(question_to_ask, k=5)

        st.success("✅ Answer Generated:")

        # Display answer
        st.markdown(response["message"]["content"])

        # Display relevant reviews
        if reviews_final:
            st.subheader("🔍 Relevant Reviews:")
            for i, r in enumerate(reviews_final):
                st.markdown(f"**Review {i+1}** (Insurer: {r['assureur']}, Rating: {r['note']}★)")
                st.markdown(f"- Test: {r['review_en']}")

st.markdown("---")
st.caption("Q&A - Insurance Review Insights (powered by RAG & Ollama)")