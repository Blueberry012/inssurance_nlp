# =====================================================
# Imports
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

# =====================================================
# Page Config
# =====================================================
st.set_page_config(layout="wide")
st.title("🔍 Semantic Search with Word2Vec & GloVe")
st.markdown("---")

# =====================================================
# Sidebar Settings
# =====================================================
st.sidebar.header("⚙️ Semantic Search Settings")

# Nombre de résultats
top_k = st.sidebar.slider("Number of results to display", 1, 20, 5)

# Liste de mots par défaut
default_words = [
    "insurance","policy","coverage","premium",
    "claim","accident","damage",
    "price","expensive","cheap",
    "service","customer","support",
    "refund","cancel","complaint",
    "good","bad","fast","slow"
]

# Saisie optionnelle de mots supplémentaires
extra_words_input = st.sidebar.text_input(
    "Add your own words (comma separated):", ""
)
extra_words = [w.strip() for w in extra_words_input.split(",") if w.strip() != ""]

# Multiselect avec tous les mots par défaut + ceux ajoutés
pca_words = st.sidebar.multiselect(
    "Select words for PCA visualization",
    options=default_words + extra_words,
    default=default_words  # tout sélectionné par défaut
)

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_clean.xlsx")
    df = df[df["type"] == "train"].reset_index(drop=True)
    df = df.dropna(subset=["avis_spacy"])
    df["avis_spacy"] = df["avis_spacy"].astype(str).str.strip()
    df = df[df["avis_spacy"].str.len() > 10].reset_index(drop=True)
    return df

df = load_data()
st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Train Word2Vec
# =====================================================
@st.cache_data
def train_word2vec(sentences):
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        epochs=10,
        seed=42
    )
    return model

sentences = [text.split() for text in df["avis_spacy"].tolist()]
w2v_model = train_word2vec(sentences)

# =====================================================
# Load GloVe (pretrained)
# =====================================================
@st.cache_data
def load_glove():
    from gensim.downloader import load as gensim_load
    glove = gensim_load("glove-wiki-gigaword-100")
    return glove

glove = load_glove()

# =====================================================
# Semantic Search Functions
# =====================================================
def get_sentence_vector(text, model):
    tokens = text.lower().split()
    vectors = []
    for word in tokens:
        if hasattr(model, "__contains__") and word in model:  # Word2Vec
            vectors.append(model[word])
        elif hasattr(model, "get_vector") and word in model:  # GloVe
            vectors.append(model.get_vector(word))
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def cosine_similarity(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)

def semantic_search(query, texts, model, top_k=5):
    query_vec = get_sentence_vector(query, model)
    if query_vec is None:
        return []
    scores = []
    for idx, text in enumerate(texts):
        doc_vec = get_sentence_vector(text, model)
        if doc_vec is None:
            continue
        similarity = cosine_similarity(query_vec, doc_vec)
        scores.append((idx, similarity))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# =====================================================
# PCA Visualization
# =====================================================
if pca_words:
    st.subheader("📊 PCA Visualization of Selected Words")
    def plot_embeddings_pca(embeddings_dict, words, title, ax, color):
        vectors = [embeddings_dict[w] for w in words if w in embeddings_dict]
        valid_words = [w for w in words if w in embeddings_dict]
        if not vectors:
            return
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)
        x = reduced[:,0]
        y = reduced[:,1]
        ax.scatter(x,y,color=color)
        for i, word in enumerate(valid_words):
            ax.annotate(word, (x[i], y[i]))
        ax.set_title(title)

    fig, axes = plt.subplots(1,2, figsize=(14,6))
    # Word2Vec
    w2v_dict = {w: w2v_model.wv[w] for w in pca_words if w in w2v_model.wv}
    plot_embeddings_pca(w2v_dict, pca_words, "Word2Vec PCA", axes[0], color="steelblue")
    # GloVe
    glove_dict = {w: glove.get_vector(w) for w in pca_words if w in glove}
    plot_embeddings_pca(glove_dict, list(glove_dict.keys()), "GloVe PCA", axes[1], color="tomato")
    st.pyplot(fig)

st.markdown("---")

# =====================================================
# Semantic Search (à la fin)
# =====================================================
st.header("🧪 Semantic Search")

# -----------------------------------------------------
# Load Test Reviews
# -----------------------------------------------------
@st.cache_data
def load_test_reviews():
    df_test = pd.read_excel("data/reviews_clean.xlsx")
    df_test = df_test[df_test["type"] == "test"].reset_index(drop=True)
    return df_test

df_test = load_test_reviews()

# -----------------------------------------------------
# User Input: choose from test or free text
# -----------------------------------------------------
selected_review = st.selectbox(
    "Select a review from the test dataset:",
    df_test["avis_en"].tolist()
)

user_input = st.text_area("Or enter your own query:", "")
input_query = user_input.strip() if user_input.strip() != "" else selected_review

# -----------------------------------------------------
# Run Semantic Search
# -----------------------------------------------------
if st.button("Search"):
    if input_query.strip() == "":
        st.warning("Please enter a query or select a review from the test dataset.")
    else:
        texts_search = df["avis_en"].fillna("").astype(str).tolist()
        texts_spacy  = df["avis_spacy"].fillna("").astype(str).tolist()

        st.subheader("Top results (Word2Vec)")
        results_w2v = semantic_search(input_query, texts_spacy, w2v_model.wv, top_k=top_k)
        for rank, (idx, score) in enumerate(results_w2v, 1):
            review_text = texts_search[idx][:200].replace("\n"," ")
            note = df["note"].iloc[idx] if "note" in df.columns else "?"
            st.markdown(f"**#{rank} (score={score:.3f}, rating={note})**  \n{review_text} ...")

        st.subheader("Top results (GloVe)")
        results_glove = semantic_search(input_query, texts_spacy, glove, top_k=top_k)
        for rank, (idx, score) in enumerate(results_glove, 1):
            review_text = texts_search[idx][:200].replace("\n"," ")
            note = df["note"].iloc[idx] if "note" in df.columns else "?"
            st.markdown(f"**#{rank} (score={score:.3f}, rating={note})**  \n{review_text} ...")

st.markdown("---")
st.caption("Semantic Search powered by Word2Vec & GloVe with PCA visualization")