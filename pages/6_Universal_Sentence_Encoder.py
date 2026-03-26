# =====================================================
# Imports
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# Page Config
# =====================================================
st.set_page_config(layout="wide")
st.title("📊 TF-IDF + Logistic Regression with Top Contributing Words")
st.markdown("---")

# =====================================================
# Sidebar Settings
# =====================================================
st.sidebar.header("⚙️ TF-IDF + Logistic Regression Settings")

max_features = st.sidebar.slider("Max Features (TF-IDF)", 1000, 10000, 5000, step=500)
max_iter = st.sidebar.slider("Max Iterations (Logistic Regression)", 100, 2000, 1000, step=100)
solver_options = ["lbfgs", "liblinear", "sag", "saga", "newton-cg"]
solver = st.sidebar.selectbox("Solver (Logistic Regression)", solver_options, index=0)

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_nlp.xlsx")
    df = df.dropna()
    df["note"] = df["note"].astype(int)
    df["label"] = df["note"] - 1
    return df

df = load_data()

# =====================================================
# Data Preview
# =====================================================
st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Train Model
# =====================================================
@st.cache_data
def train_model(df, max_features, max_iter, solver):

    X = df["avis_cor"]
    y = df["note"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
        stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            max_df=0.5,
            min_df=5
        )),
        ("logreg", LogisticRegression(
            max_iter=max_iter,
            C=0.5,
            solver=solver,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

model, X_test, y_test, y_pred = train_model(df, max_features, max_iter, solver)

# =====================================================
# Model Metrics
# =====================================================
st.header("📈 Model Performance")

acc  = accuracy_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
f1   = f1_score(y_test, y_pred, average="macro")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("MAE", f"{mae:.3f}")
col3.metric("RMSE", f"{rmse:.3f}")
col4.metric("F1 Score", f"{f1:.3f}")

st.markdown("---")

# =====================================================
# Confusion Matrix
# =====================================================
st.subheader("📊 Confusion Matrix")

def plot_confusion_matrix_compact(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    labels_star = [f"{i}★" for i in labels]

    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels_star,
        yticklabels=labels_star,
        ax=ax,
        cmap="Blues",
        cbar=False
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig

fig = plot_confusion_matrix_compact(y_test, y_pred)
st.pyplot(fig, use_container_width=False)

# =====================================================
# Important Words per Class
# =====================================================
st.header("🔑 Important Words per Class")

tfidf = model.named_steps["tfidf"]
logreg = model.named_steps["logreg"]
feature_names = tfidf.get_feature_names_out()
top_n_words = 10
cols = st.columns(len(logreg.classes_))

for i, class_label in enumerate(logreg.classes_):
    with cols[i]:
        st.markdown(f"⭐ Class {class_label}")
        top_indices = np.argsort(logreg.coef_[i])[-top_n_words:]
        for idx in top_indices[::-1]:
            word = feature_names[idx]
            coef = logreg.coef_[i][idx]
            st.caption(f"{word} → {coef:.6f}")

st.markdown("---")

# =====================================================
# Test Your Own Review / Test Dataset
# =====================================================
st.header("🧪 Test Your Own Review or Select from Test Reviews")

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

user_input = st.text_area("Or Enter your own review here:", "")
input_review = user_input.strip() if user_input.strip() != "" else selected_review

# =====================================================
# Function to get top contributing words
# =====================================================
def get_top_contributing_words(model, text, top_n=10):
    tfidf = model.named_steps["tfidf"]
    logreg = model.named_steps["logreg"]

    # ⚡ transformer via pipeline
    X_tfidf = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()

    pred_class = model.predict([text])[0]
    class_index = list(logreg.classes_).index(pred_class)
    coefs = logreg.coef_[class_index]
    tfidf_values = X_tfidf.toarray()[0]

    contributions = tfidf_values * coefs
    top_indices = np.argsort(contributions)[-top_n:]

    results = []
    for idx in top_indices[::-1]:
        if contributions[idx] > 0:
            results.append((feature_names[idx], contributions[idx]))
    return results

# =====================================================
# Predict and display top words
# =====================================================
if st.button("Predict"):
    if input_review.strip() != "":
        pred = model.predict([input_review])[0]
        st.success(f"⭐ Predicted Rating: {pred}")

        st.subheader("Top contributing words")
        st.write("----------------------------------------")
        top_words = get_top_contributing_words(model, input_review, top_n=10)
        for word, score in top_words:
            st.text(f"{word:<25} : {score:.4f}")
    else:
        st.warning("Please enter a review or select one from the test dataset")

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.caption("TF-IDF + Logistic Regression Model with Top Contributing Words")