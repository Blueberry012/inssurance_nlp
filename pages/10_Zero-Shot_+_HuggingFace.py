# =====================================================
# Zero-Shot Classification Streamlit (with custom subjects)
# =====================================================

import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(layout="wide")
st.title("📝 Zero-Shot Classification for Insurance Reviews")
st.markdown("---")

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_reviews():
    df = pd.read_excel("data/reviews_nlp.xlsx")
    df = df[['avis_en','note']].dropna()
    df['note'] = df['note'].astype(int)
    return df

df = load_reviews()
st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Zero-Shot Pipeline
# =====================================================
@st.cache_resource
def load_classifier():
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1  # CPU, mettre 0 pour GPU
    )
    return classifier

classifier = load_classifier()

# =====================================================
# Default Categories
# =====================================================
default_categories = [
    "Pricing",
    "Coverage",
    "Enrollment",
    "Customer Service",
    "Claims Processing",
    "Cancellation",
    "Other"
]

# =====================================================
# Sidebar Settings
# =====================================================
st.sidebar.header("⚙️ Zero-Shot Settings")

# Multiselect pour choisir parmi les catégories par défaut
selected_categories = st.sidebar.multiselect(
    "Select default categories:",
    options=default_categories,
    default=default_categories
)

# Zone pour ajouter ses propres catégories
extra_categories_input = st.sidebar.text_input(
    "Add your own subjects (comma-separated):", ""
)
extra_categories = [c.strip() for c in extra_categories_input.split(",") if c.strip()]

# Combinaison des catégories
categories = selected_categories + extra_categories

# =====================================================
# User Input
# =====================================================
st.header("🧪 Classify a Review")

selected_review = st.selectbox(
    "Select a review from the dataset:",
    df['avis_en'].tolist()
)
user_review = st.text_area("Or enter your own review:", "")

input_review = user_review.strip() if user_review.strip() != "" else selected_review

# =====================================================
# Prediction Function
# =====================================================
def predict_category(text, candidate_labels):
    result = classifier(
        text,
        candidate_labels=candidate_labels,
        multi_label=False
    )
    return result['labels'][0], result['scores'][0]

# =====================================================
# Run Prediction
# =====================================================
if st.button("Predict Category"):
    if not input_review:
        st.warning("Please enter a review or select one from the dataset.")
    elif not categories:
        st.warning("Please select or add at least one category.")
    else:
        category, score = predict_category(input_review, categories)
        st.subheader("✅ Predicted Category")
        st.markdown(f"- **Category:** {category}")
        st.markdown(f"- **Confidence Score:** {score:.3f}")
        st.subheader("📝 Review Text")
        st.write(input_review)