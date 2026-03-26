# =====================================================
# TF-IDF + XGBoost Streamlit App
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(layout="wide")
st.title("📊 TF-IDF + XGBoost for Sentiment Analysis with Top Contributing Words")
st.markdown("---")

# =====================================================
# Sidebar Settings
# =====================================================
st.sidebar.header("⚙️ Model Settings")

max_features = st.sidebar.slider("Max Features (TF-IDF)", 1000, 10000, 5000, step=500)
ngram_range = st.sidebar.slider("N-gram range (max n)", 1, 3, 2, step=1)
max_df = st.sidebar.slider("Max DF (TF-IDF)", 0.1, 1.0, 0.5, step=0.05)
min_df = st.sidebar.slider("Min DF (TF-IDF)", 1, 10, 5, step=1)

max_depth = st.sidebar.slider("Max Depth (XGBoost)", 3, 10, 6, step=1)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
n_estimators = st.sidebar.slider("Number of Trees", 100, 500, 300, step=50)

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_sen.xlsx")
    df = df[['avis_cor', 'note']].dropna()
    
    # Map sentiment: negative=0, neutral=1, positive=2
    def map_sentiment(note):
        if note in [1, 2]:
            return 0
        elif note == 3:
            return 1
        else:
            return 2
    df['sentiment'] = df['note'].apply(map_sentiment)
    return df

df = load_data()
st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Train/Test Split
# =====================================================
X = df['avis_cor']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# =====================================================
# TF-IDF + XGBoost Training
# =====================================================
@st.cache_data
def train_model(X_train, y_train, max_features, ngram_range, max_df, min_df,
                max_depth, learning_rate, n_estimators):

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_range),
        max_df=max_df,
        min_df=min_df
    )
    X_train_tfidf = tfidf.fit_transform(X_train)

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_tfidf, y_train)
    return tfidf, model

tfidf, xgb_model = train_model(
    X_train, y_train,
    max_features, ngram_range, max_df, min_df,
    max_depth, learning_rate, n_estimators
)

# =====================================================
# Evaluate Model
# =====================================================
X_test_tfidf = tfidf.transform(X_test)
y_pred = xgb_model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
f1_macro = f1_score(y_test, y_pred, average="macro")

st.header("📈 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("MAE", f"{mae:.3f}")
col3.metric("RMSE", f"{rmse:.3f}")
col4.metric("F1 Macro", f"{f1_macro:.3f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

# =====================================================
# Confusion Matrix
# =====================================================
st.subheader("📊 Confusion Matrix")

def plot_confusion_matrix_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels_star = ["negative", "neutral", "positive"]
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_star,
                yticklabels=labels_star, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=False)

plot_confusion_matrix_cm(y_test, y_pred)

# =====================================================
# Top Words Per Sentiment Class
# =====================================================
st.header("🔑 Top Words Per Sentiment Class")

def get_top_words_per_class(X_train, y_train, tfidf, model, labels, n_words=10):
    feature_names = tfidf.get_feature_names_out()
    top_words = {}
    for class_idx, class_name in enumerate(labels):
        mask = y_train.values == class_idx
        class_tfidf = tfidf.transform(X_train[mask])
        avg_tfidf = np.array(class_tfidf.mean(axis=0)).flatten()
        top_indices = avg_tfidf.argsort()[-n_words:][::-1]
        top_words[class_name] = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
    return top_words

labels = ["negative", "neutral", "positive"]
top_words_per_class = get_top_words_per_class(X_train, y_train, tfidf, xgb_model, labels)

cols = st.columns(3)
for i, class_name in enumerate(labels):
    with cols[i]:
        st.markdown(f"⭐ {class_name.upper()}")
        for word, score in top_words_per_class[class_name]:
            st.caption(f"{word} → {score:.4f}")

# =====================================================
# Test Your Own Review
# =====================================================
st.header("🧪 Test a Review")

selected_review = st.selectbox("Select a review from the dataset:", X_test.tolist())
user_input = st.text_area("Or enter your own review:", "")
input_review = user_input.strip() if user_input.strip() else selected_review

def get_important_words_for_review(review, tfidf, xgb_model, n_words=10):
    feature_names = tfidf.get_feature_names_out()
    feature_importances = xgb_model.feature_importances_
    vec = tfidf.transform([review])
    tfidf_scores = np.array(vec.todense()).flatten()
    combined_scores = tfidf_scores * feature_importances
    top_indices = combined_scores.argsort()[-n_words:][::-1]
    important_words = [(feature_names[i], combined_scores[i]) for i in top_indices if combined_scores[i] > 0]
    return important_words

def predict_review(review):
    vec = tfidf.transform([review])
    pred = xgb_model.predict(vec)[0]
    return labels[pred]

if st.button("Predict"):
    if input_review:
        pred = predict_review(input_review)
        st.success(f"⭐ Predicted Sentiment: {pred}")

        st.subheader("Top Contributing Words")
        top_words = get_important_words_for_review(input_review, tfidf, xgb_model)
        if top_words:
            for word, score in top_words:
                st.text(f"{word:<25} : {score:.4f}")
        else:
            st.text("(No contributing words found in vocabulary)")
    else:
        st.warning("Please enter a review or select one from the dataset.")

st.markdown("---")
st.caption("TF-IDF + XGBoost Sentiment Analysis with Top Contributing Words")