# =====================================================
# Imports
# =====================================================

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# Page Config
# =====================================================

st.set_page_config(layout="wide")
st.title("🤖 DistilBERT + LoRA Sentiment Analysis")
st.markdown("---")

# =====================================================
# Sidebar Settings
# =====================================================

st.sidebar.header("⚙️ Model Settings")
max_length = st.sidebar.slider("Max Token Length", 64, 256, 128, step=16)

# =====================================================
# Load Data
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_sen.xlsx")
    df = df[['avis_cor','note']].dropna()
    return df

df = load_data()

# =====================================================
# Map sentiment
# =====================================================

def map_sentiment(note):
    if note in [1,2]:
        return 0
    elif note == 3:
        return 1
    else:
        return 2

df['sentiment'] = df['note'].apply(map_sentiment)
X = df['avis_cor'].tolist()
y = df['sentiment'].tolist()

# =====================================================
# Split Data
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

# =====================================================
# Load BERT + LoRA Model
# =====================================================

@st.cache_resource
def load_bert_model():
    model_path = "./model/distilbert_lora_sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_bert_model()
labels = ["negative","neutral","positive"]

# =====================================================
# Prediction Function
# =====================================================

def predict_review(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return pred

# =====================================================
# Evaluate Model on Full Test Set
# =====================================================

st.header("🧪 Evaluate Model on Full Test Set")

st.info("Running predictions on the full test set...")
preds = [predict_review(text) for text in X_test]

# Classification Report
st.subheader("📊 Classification Report")
report = classification_report(y_test, preds, target_names=labels, output_dict=True)
st.dataframe(pd.DataFrame(report).T)

# Compact Confusion Matrix
st.subheader("📈 Confusion Matrix")
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
    cbar=False
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# =====================================================
# Test a Single Review (At the End)
# =====================================================

st.header("🧪 Test a Single Review")
st.markdown("Select a review from the test dataset or enter your own:")

@st.cache_data
def load_test_reviews():
    df_test = pd.read_excel("data/reviews_clean.xlsx")
    df_test = df_test[df_test["type"] == "test"].reset_index(drop=True)
    return df_test

df_test = load_test_reviews()

selected_review = st.selectbox(
    "Select a review from the test dataset:",
    df_test["avis_en"].tolist()
)

user_input = st.text_area("Or enter your own review here:")

input_review = user_input.strip() if user_input.strip() != "" else selected_review

if st.button("Predict Sentiment", key="single_review"):
    if input_review.strip() != "":
        pred = predict_review(input_review)
        st.success(f"⭐ Predicted Sentiment: {labels[pred]}")
    else:
        st.warning("Please enter a review or select one from the test dataset")

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.caption("DistilBERT + LoRA Sentiment Analysis")