import streamlit as st

# =====================================================
# Page Config
# =====================================================

st.set_page_config(layout="wide")

# Title
st.title("🧠 NLP Review Analysis System")
st.subheader("Supervised Learning & Text Processing Application")
st.image("image/insurance.jpeg", width=750, clamp=True)

st.markdown("---")

# =====================================================
# Project Description
# =====================================================

st.header("📌 Project Overview")

st.markdown("""
This project focuses on **Natural Language Processing (NLP)** applied to **customer reviews analysis**.

### 🎯 Main Goal
Build supervised learning models to automatically analyze reviews and predict:

- ⭐ Number of stars  
- 😊 Sentiment (positive, neutral, negative)  
- 🏷️ Review category (Pricing, Customer Service, Claims, etc.)

### 💡 Main Hypothesis
Customer reviews contain enough textual information to:

- Predict ratings  
- Detect sentiment  
- Identify the main subject  

This means we can extract valuable insights **purely from text data**.
""")

# =====================================================
# Input / Output Model
# =====================================================

st.header("🔄 Input / Output Model")

st.markdown("""
### ✅ Input
- A textual review written by a user

### ✅ Output
- ⭐ Predicted star rating (1–5)
- 😊 Predicted sentiment
- 🏷️ Predicted category (e.g., Pricing, Coverage, Customer Service)

The system relies on NLP models to transform text into meaningful predictions.
""")

# =====================================================
# Models Used
# =====================================================

st.header("🧠 Models & Approaches")

st.markdown("""
### 📊 Classical Models
- TF-IDF + Machine Learning (Logistic Regression, Random Forest)

---

### 🚀 Advanced Models
- Word Embeddings (Word2Vec, GloVe)
- Deep Learning Models (LSTM, CNN)
- Transformer Models (BERT, Hugging Face)

---

### 🤖 Additional Techniques
- Zero-shot classification (for categories)
- Text preprocessing & cleaning
- Embedding visualization
""")

# =====================================================
# Evaluation Strategy
# =====================================================

st.header("📈 Evaluation Methodology")

st.markdown("""
To evaluate model performance, we use:

### Metrics
- Accuracy
- F1-score
- Precision / Recall

---

### Evaluation Strategy
- Compare multiple models
- Analyze prediction errors
- Evaluate performance per class (stars, sentiment, categories)

This ensures robustness and reliability of predictions.
""")

# =====================================================
# Features of the System
# =====================================================

st.header("✨ System Features")

st.markdown("""
✔ Interactive prediction interface  
✔ Real-time text analysis  
✔ Sentiment & category detection  
✔ Model comparison dashboard  
✔ Explainability of predictions  
✔ Review search & filtering  
✔ Visualization of results  
""")

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.caption("Supervised NLP Project - Streamlit Application")