import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json

# Load tokenizer from tokenizer_config.json
with open("tokenizer_config.json", "r", encoding="utf-8") as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load the trained Keras model (H5 format)
model = tf.keras.models.load_model("sms_spam_model.h5")

# Parameters
max_length = 100  # Must match training max_length

# Streamlit UI
st.title("📱 SMS Spam Detection")
st.write("Enter a message to check if it's spam or not:")

# Text input
user_input = st.text_area("Your message here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Tokenize and pad input message
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Predict probability
        prediction = model.predict(padded, verbose=0)[0][0]
        label = "🚫 Spam" if prediction > 0.5 else "✅ Ham"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Display result with confidence
        st.subheader("Result:")
        st.write(f"**{label}** (Confidence: `{confidence * 100:.2f}%`)")

# Footer
st.markdown("---")
