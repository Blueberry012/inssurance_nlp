# =====================================================
# Imports
# =====================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# =====================================================
# Page Config
# =====================================================
st.set_page_config(layout="wide")
st.title("📊 N-Grams & Word Cloud")
st.markdown("---")

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_clean.xlsx")
    df = df.dropna(subset=["avis_spacy"])
    return df

df = load_data()

st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Sidebar Settings
# =====================================================
st.sidebar.header("⚙️ N-Gram Settings")

ngram_min = st.sidebar.number_input("N-Gram Min", 1, 5, 1, step=1)
ngram_max = st.sidebar.number_input("N-Gram Max", 1, 5, 1, step=1)
max_features = st.sidebar.slider("Max Features (Top Words)", 5, 50, 20, step=5)
colormap = st.sidebar.selectbox("WordCloud Colormap", ["viridis", "plasma", "coolwarm", "magma", "cividis"])

# =====================================================
# Prepare Text Data
# =====================================================
text_data = df["avis_spacy"].dropna().astype(str)

# =====================================================
# CountVectorizer - Top Words
# =====================================================
vectorizer = CountVectorizer(
    max_features=max_features,
    ngram_range=(ngram_min, ngram_max)
)

X = vectorizer.fit_transform(text_data)
words = vectorizer.get_feature_names_out()
counts = X.toarray().sum(axis=0)

# =====================================================
# Display Bar Chart
# =====================================================
st.subheader("📊 Top Frequent Words / N-Grams")
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(words, counts, color="skyblue")
ax.set_xticklabels(words, rotation=45, ha="right")
ax.set_title(f"Top {max_features} frequent words / N-grams ({ngram_min}-{ngram_max})")
st.pyplot(fig)

# =====================================================
# WordCloud
# =====================================================
st.subheader("🌐 Word Frequency Map")
text = " ".join(text_data)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    colormap=colormap
).generate(text)

fig_wc, ax_wc = plt.subplots(figsize=(15,7))
ax_wc.imshow(wordcloud, interpolation="bilinear")
ax_wc.axis("off")
ax_wc.set_title("Word Frequency Map")
st.pyplot(fig_wc)