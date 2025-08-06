import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import nltk
from nltk.corpus import stopwords
import string

# Download stopwords quietly; ignores if already downloaded
nltk.download('stopwords', quiet=True)

# Load tokenizer from JSON (cached so it doesn't reload on every rerun)
@st.cache_resource
def load_tokenizer():
    with open("tokenizer_config.json", "r", encoding='utf-8') as f:
        tokenizer_json = f.read()
    return tokenizer_from_json(tokenizer_json)

tokenizer = load_tokenizer()

# Load model (cached similarly)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sms_spam_model.h5")

model = load_model()

# Constants, must match those used during training
max_length = 100
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """
    Remove punctuation, convert to lowercase, and remove stopwords.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.lower().split() if word not in stop_words])

# Streamlit UI setup
st.set_page_config(page_title="SMS Spam Detection", layout="centered")
st.title("📱 SMS Spam Detection App")
st.markdown("Enter an SMS message below to check whether it's **spam** or **ham**.")

user_input = st.text_area("✉️ Enter SMS message:", height=150)

def classify_message(message):
    cleaned = preprocess_text(message)
    if not cleaned.strip():
        return None  # Return None if no valid text remains after preprocessing
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    return prediction

if st.button("🔍 Check Message"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        pred = classify_message(user_input)
        if pred is None:
            st.warning("Message is empty after preprocessing (too many stopwords or only punctuation). Please enter a more descriptive message.")
        elif pred >= 0.5:
            st.error("🚫 This message is **Spam**.")
        else:
            st.success("✅ This message is **Ham** (Not Spam).")

st.markdown("---")
