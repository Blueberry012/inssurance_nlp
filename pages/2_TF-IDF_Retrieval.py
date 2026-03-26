# =====================================================
# Imports
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# Page Config
# =====================================================
st.set_page_config(layout="wide")
st.title("📄 TF-IDF Retrieval — Top-K Similar Reviews")
st.markdown("---")

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_clean.xlsx")
    df = df.dropna(subset=["avis_spacy"])
    df["avis_spacy"] = df["avis_spacy"].astype(str).str.strip()
    df = df[df["avis_spacy"].str.len() > 10]
    df["type"] = df["type"].astype(str).str.lower().str.strip()
    df["produit"] = df["produit"].astype(str).str.strip()
    return df

df = load_data()

st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Category Merging
# =====================================================
MERGE_MAP = {
    "moto"       : "auto-moto",
    "auto"       : "auto-moto",
    "sante"      : "sante-prevoyance-vie",
    "prevoyance" : "sante-prevoyance-vie",
    "vie"        : "sante-prevoyance-vie",
}
df["produit"] = df["produit"].apply(lambda p: MERGE_MAP.get(p, p))

# Filter products with >= 500 reviews in train
MIN_SAMPLES = 500
counts_train = df[df["type"] == "train"]["produit"].value_counts()
produits_valides = counts_train[counts_train >= MIN_SAMPLES].index.tolist()
train_df = df[(df["type"] == "train") & (df["produit"].isin(produits_valides))].copy().reset_index(drop=True)
test_df  = df[(df["type"] == "test")  & (df["produit"].isin(produits_valides))].copy().reset_index(drop=True)

produits = sorted(produits_valides)
produit2id = {p: i for i, p in enumerate(produits)}
id2produit = {i: p for p, i in produit2id.items()}

train_df["label"] = train_df["produit"].map(produit2id)
test_df["label"]  = test_df["produit"].map(produit2id)

# =====================================================
# Undersampling
# =====================================================
min_size = train_df["label"].value_counts().min()
train_bal = (train_df.groupby("label", group_keys=False)
                     .apply(lambda x: x.sample(min_size, random_state=42))
                     .reset_index(drop=True))

# =====================================================
# TF-IDF Vectorization
# =====================================================
vectorizer = TfidfVectorizer(
    max_features=5000,
    max_df=0.90,
    min_df=2,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_bal["avis_spacy"].tolist())
X_test  = vectorizer.transform(test_df["avis_spacy"].tolist())
y_train = train_bal["label"].values
y_test  = test_df["label"].values
feature_names = np.array(vectorizer.get_feature_names_out())

st.subheader("TF-IDF Undersampled")

# =====================================================
# Retrieval Functions
# =====================================================
def topk_indices(doc_matrix, query_matrix, k: int) -> np.ndarray:
    sims = (doc_matrix @ query_matrix.T).toarray().T
    k = min(k, sims.shape[1])
    part = np.argpartition(sims, -k, axis=1)[:, -k:]
    row = np.arange(sims.shape[0])[:, None]
    order = np.argsort(sims[row, part], axis=1)[:, ::-1]
    return part[row, order]

class_weights = {p_id: int((y_train == p_id).sum()) for p_id in id2produit}

def majority_label(lbls: list) -> int:
    counts = {p_id: lbls.count(p_id) for p_id in id2produit}
    max_count = max(counts.values())
    candidates = [p_id for p_id, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda p: class_weights[p])

# =====================================================
# TF-IDF Retrieval
# =====================================================
K = st.sidebar.slider("Top-K retrieval", 1, 20, 11)
topk = topk_indices(X_train, X_test, k=K)

y_pred = np.array([
    majority_label([y_train[i] for i in topk[qi]])
    for qi in range(len(y_test))
])

overall = float((y_pred == y_test).mean())
st.metric(f"Overall Precision@{K}", f"{overall:.4f}")

# =====================================================
# Precision per Product — Bar Chart
# =====================================================
precisions = []
labels_viz = []
for p_id, p_name in id2produit.items():
    mask = (y_test == p_id)
    if mask.sum() == 0:
        continue
    precisions.append(float((y_pred[mask] == p_id).mean()))
    labels_viz.append(p_name)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2ecc71" if p >= overall else "#e74c3c" for p in precisions]
bars = ax.barh(labels_viz, precisions, color=colors, alpha=0.85)
ax.axvline(overall, color="navy", linestyle="--", linewidth=1.5,
           label=f"Overall P@{K} = {overall:.3f}")
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
ax.set_xlim(0, 1.1)
ax.set_xlabel(f"Precision@{K}")
ax.set_title(f"TF-IDF Retrieval — Precision@{K} per Product", fontsize=13)
ax.legend()
st.pyplot(fig)

# =====================================================
# Top Words per Category — en colonnes
# =====================================================
st.subheader("🔑 Top Words per Product")

top_n = st.sidebar.slider("Top-N words per product", 5, 15, 10)

# Créer une colonne par produit
cols = st.columns(len(id2produit))

for i, (class_id, class_name) in enumerate(id2produit.items()):
    mask = (y_train == class_id)
    if mask.sum() == 0:
        continue
    mean_tfidf = X_train[mask].mean(axis=0).A1
    top_idx = np.argsort(mean_tfidf)[-top_n:][::-1]
    top_words = [(feature_names[idx], mean_tfidf[idx]) for idx in top_idx]

    with cols[i]:
        st.markdown(f"**{class_name.upper()}**")
        for w, score in top_words:
            st.caption(f"{w:<15} → {score:.4f}")

# =====================================================
# Example Queries / Test Set
# =====================================================
st.subheader(f"🧪 Example Queries — Top-{K} Retrieval")
example_per_product = 3

for p_id, p_name in id2produit.items():
    indices = np.where(y_test == p_id)[0][:example_per_product]
    if len(indices) == 0:
        continue
    st.markdown(f"### {p_name.upper()}")
    for rank, qi in enumerate(indices, 1):
        lbls      = [y_train[i] for i in topk[qi]]
        ret_lbls  = [id2produit[l] for l in lbls]
        predicted = id2produit[y_pred[qi]]
        correct   = "✅" if y_pred[qi] == y_test[qi] else "❌"
        query_text = test_df['avis_en'].iloc[qi].replace("\n", " ")
        st.markdown(f"**Query #{rank}:** {query_text}")
        st.markdown(f"  - Top-{K} retrieved: {ret_lbls}")
        st.markdown(f"  - Predicted: {predicted} {correct}")

# =====================================================
# Predict product for a test review and show top words
# =====================================================
st.header("🧪 Predict Product for a Review (TF-IDF weighted)")
def load_test_reviews():
    df_test = pd.read_excel("data/reviews_clean.xlsx")
    df_test = df_test[df_test["type"] == "test"].reset_index(drop=True)
    return df_test

df_test = load_test_reviews()

# Sélecteur ou saisie
selected_review = st.selectbox(
    "Select a review from the test dataset:",
    df_test["avis_en"].tolist()
)
user_input = st.text_area("Or enter your own review:", "")
input_review = user_input.strip() if user_input.strip() != "" else selected_review

top_n_words = st.sidebar.slider("Top contributing words", 5, 10, 7)

if st.button("Predict Product"):
    if input_review.strip() != "":
        # TF-IDF de la review
        query_vec = vectorizer.transform([input_review]).toarray()[0]

        # Calculer un score par produit: somme des TF-IDF de mots présents dans ce produit
        product_scores = {}
        product_top_words = {}

        for class_id, class_name in id2produit.items():
            mask = (y_train == class_id)
            mean_tfidf = X_train[mask].mean(axis=0).A1
            # score = somme des TF-IDF du query * moyenne du produit
            score_per_word = query_vec * mean_tfidf
            total_score = score_per_word.sum()
            product_scores[class_name] = total_score

            # mots les plus contributifs
            top_indices = score_per_word.argsort()[-top_n_words:][::-1]
            product_top_words[class_name] = [(feature_names[i], score_per_word[i]) for i in top_indices if score_per_word[i] > 0]

        # Produit prédit = celui avec le score max
        predicted_product = max(product_scores, key=product_scores.get)
        st.success(f"⭐ Predicted Product: {predicted_product}")

        # Afficher top contributing words pour ce produit
        st.subheader(f"Top contributing words for prediction ({predicted_product})")
        for word, score in product_top_words[predicted_product]:
            st.text(f"{word:<15} → {score:.4f}")

        # Optionnel : afficher tous les scores par produit
        st.subheader("Scores by product")
        for p, s in product_scores.items():
            st.text(f"{p:<30} → {s:.4f}")
    else:
        st.warning("Please enter a review or select one from the test dataset.")

st.markdown("---")
st.caption("TF-IDF Retrieval")