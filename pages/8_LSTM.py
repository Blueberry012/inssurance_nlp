# =====================================================
# Imports
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# =====================================================
# Page Config
# =====================================================
st.set_page_config(layout="wide")
st.title("🧠 Sentiment Analysis with Bidirectional LSTM")
st.markdown("---")

# =====================================================
# Sidebar Settings
# =====================================================
st.sidebar.header("⚙️ Model Settings")
max_vocab = st.sidebar.slider("Max Vocabulary Size", 1000, 20000, 10000, step=1000)
max_len = st.sidebar.slider("Max Sequence Length", 50, 200, 100, step=10)
embedding_dim = st.sidebar.slider("Embedding Dimension", 50, 300, 100, step=50)
lstm_units = st.sidebar.slider("LSTM Units", 32, 128, 64, step=16)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.3, step=0.05)
epochs = st.sidebar.slider("Epochs", 1, 20, 5, step=1)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, step=16)

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data/reviews_sen.xlsx")
    df = df[['avis_cor','note']].dropna()
    # Map notes to sentiment classes
    def map_sentiment(note):
        if note == 1:
            return 0  # negative
        elif note in [2,3,4]:
            return 1  # neutral
        else:
            return 2  # positive
    df['sentiment'] = df['note'].apply(map_sentiment)
    return df

df = load_data()
st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# =====================================================
# Train/Test Split + Tokenization
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    df['avis_cor'].astype(str), df['sentiment'],
    test_size=0.1, random_state=42, stratify=df['sentiment']
)

tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# Compute class weights
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weights_dict = dict(enumerate(class_weights_array))

# =====================================================
# Build and Train Model
# =====================================================
@st.cache_resource
def build_train_model():
    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=embedding_dim),
        Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_train_model()
st.write("Training Bidirectional LSTM model...")
history = model.fit(
    X_train_pad, y_train_cat,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weights_dict,
    verbose=1
)

# =====================================================
# Evaluate Model
# =====================================================
y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = np.mean(y_pred == y_test)
st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("F1 Score (Macro)", f"{f1_score(y_test, y_pred, average='macro'):.3f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred, target_names=["negative","neutral","positive"]))

# Confusion matrix
st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["negative","neutral","positive"],
            yticklabels=["negative","neutral","positive"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig, use_container_width=False)

# =====================================================
# Test Your Own Review
# =====================================================
st.header("🧪 Test a Review")
selected_review = st.selectbox("Select a review from the dataset:", X_test.tolist())
user_input = st.text_area("Or enter your own review:", "")
input_review = user_input.strip() if user_input.strip() else selected_review

def predict_review(review):
    seq = tokenizer.texts_to_sequences([review])
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = np.argmax(model.predict(pad), axis=1)[0]
    return ["negative","neutral","positive"][pred]

if st.button("Predict"):
    if input_review:
        pred = predict_review(input_review)
        st.success(f"⭐ Predicted Sentiment: {pred}")
    else:
        st.warning("Please enter a review or select one from the dataset.")

st.markdown("---")
st.caption("Bidirectional LSTM Sentiment Analysis")