# =====================================================
# Imports
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import gensim.downloader as api

# =====================================================
# Classe Gensim Interface (préchargement en mémoire)
# =====================================================
class gensim_interface:
    """
    Interface légère pour charger des embeddings depuis Gensim
    sans stocker de fichiers volumineux sur GitHub.
    """
    def __init__(self, embeddingName):
        print(f"Téléchargement du modèle '{embeddingName}' depuis Gensim si nécessaire...")
        self.embeddingVectors = api.load(embeddingName)  # téléchargement automatique
        self.vectors = self.embeddingVectors.vectors

    def isVec(self, word):
        return word in self.embeddingVectors

    def getVec(self, word):
        return self.embeddingVectors[word]

    def nbDims(self):
        return self.embeddingVectors.vector_size

# =====================================================
# Page Config
# =====================================================
st.set_page_config(layout="wide")
st.title("💡 Insurance Review Recommendation with Word Embeddings")
st.markdown("---")

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_clean.xlsx")
    df = df.dropna(subset=["avis_spacy"])
    df["avis_spacy"] = df["avis_spacy"].astype(str).str.strip()
    return df

df = load_data()
st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)

# =====================================================
# Prepare Train/Test Split
# =====================================================
emb = gensim_interface('glove-wiki-gigaword-100')

grouped = df.groupby('assureur')['avis_spacy'].apply(
    lambda x: " ".join(x)
).reset_index()

X_train02, X_test02 = train_test_split(
    grouped,
    test_size=0.1,
    random_state=42
)
X_train02 = X_train02.reset_index(drop=True)
X_test02 = X_test02.reset_index(drop=True)

# =====================================================
# Embedding Functions
# =====================================================
def get_document_embedding(text, emb):
    words = text.split()
    vectors = [emb.getVec(w) for w in words if emb.isVec(w)]
    if not vectors:
        return np.zeros(emb.nbDims())
    return np.mean(vectors, axis=0)

@st.cache_data
def compute_embeddings(X_train, X_test, _emb):
    train_embeddings = np.array([get_document_embedding(t, _emb) for t in X_train['avis_spacy']])
    test_embeddings  = np.array([get_document_embedding(t, _emb) for t in X_test['avis_spacy']])
    sim_matrix = cosine_similarity(test_embeddings, train_embeddings)
    return train_embeddings, test_embeddings, sim_matrix

train_embeddings, test_embeddings, similarity_matrix_m02 = compute_embeddings(X_train02, X_test02, emb)

# =====================================================
# Explainability: Top Matching Words
# =====================================================
def explain_similarity_words(query_text, similar_text, emb, top_k_words=8, threshold=0.5):
    query_words = list(set(query_text.split()))
    similar_words = list(set(similar_text.split()))
    matched_pairs = []
    for w1 in query_words:
        if not emb.isVec(w1):
            continue
        v1 = emb.getVec(w1)
        for w2 in similar_words:
            if not emb.isVec(w2):
                continue
            v2 = emb.getVec(w2)
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            if sim > threshold:
                matched_pairs.append((w1, w2, sim))
    matched_pairs = sorted(matched_pairs, key=lambda x: x[2], reverse=True)
    return matched_pairs[:top_k_words]

# =====================================================
# Recommendation Function
# =====================================================
def recommend_insurer(assureur_query, top_k=3, top_n_words=5):
    X_test_reset = X_test02.reset_index(drop=True)
    X_train_reset = X_train02.reset_index(drop=True)

    if assureur_query not in X_test_reset['assureur'].values:
        st.warning("Insurance company not in test set, using first test entry as fallback")
        query_idx = 0
    else:
        query_idx = X_test_reset.index[X_test_reset['assureur'] == assureur_query][0]

    sims = similarity_matrix_m02[query_idx]
    top_indices = np.argsort(sims)[::-1][:top_k]
    query_text = X_test_reset.iloc[query_idx]['avis_spacy']

    st.subheader("🔎 Top Recommendations")
    cols = st.columns(top_k)

    for rank, (idx, col) in enumerate(zip(top_indices, cols), start=1):
        rec_assureur = X_train_reset.iloc[idx]['assureur']
        score = sims[idx]
        rec_text = X_train_reset.iloc[idx]['avis_spacy']
        matched_words = explain_similarity_words(query_text, rec_text, emb, top_k_words=top_n_words)

        with col:
            st.markdown(f"**{rank}. {rec_assureur}**")
            st.markdown(f"Similarity: {score:.3f}")
            st.markdown("Top Matching Words:")
            if matched_words:
                st.write([w1 for w1, w2, sim in matched_words])
            else:
                st.write("No strong semantic match found.")

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Number of Recommendations", 1, 10, 3)
top_n_words = st.sidebar.slider("Top Matching Words per Recommendation", 1, 10, 5)

selected_assureur = st.sidebar.selectbox(
    "Select Insurance Company from Test Set",
    X_test02['assureur'].tolist()
)

if st.sidebar.button("🚀 Recommend"):
    recommend_insurer(selected_assureur, top_k=top_k, top_n_words=top_n_words)
