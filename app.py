import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# Load tokenizer from JSON
with open("tokenizer_config.json", "r") as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Load the trained model
model = tf.keras.models.load_model("sms_spam_model.h5")

# Constants (must match training)
max_length = 100

# Setup stopwords and punctuation for preprocessing
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase and remove stopwords
    return ' '.join([word for word in text.lower().split() if word not in stop_words])

# Streamlit UI setup
st.set_page_config(page_title="SMS Spam Detection", layout="centered")
st.title("📱 SMS Spam Detection App")
st.markdown("Enter an SMS message below to check whether it's **spam** or **ham**.")

user_input = st.text_area("✉️ Enter SMS message:", height=150)

def classify_message(message):
    cleaned = preprocess_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)[0][0]
    return prediction

if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        pred = classify_message(user_input)
        if pred >= 0.5:
            st.error("🚫 This message is **Spam**.")
        else:
            st.success("✅ This message is **Ham** (Not Spam).")

st.markdown("---")
