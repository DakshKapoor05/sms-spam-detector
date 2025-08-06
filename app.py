import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json

# Load tokenizer from tokenizer_config.json
with open("tokenizer_config.json", "r") as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load the trained Keras model
model = tf.keras.models.load_model("sms_spam_model.keras")

# Parameters
max_length = 100  # same as used during training

# UI
st.title("📱 SMS Spam Detection")
st.write("Enter a message to check if it's spam or not:")

# Text input
user_input = st.text_area("Your message here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Predict
        prediction = model.predict(padded)[0][0]
        label = "🚫 Spam" if prediction > 0.5 else "✅ Ham"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Result
        st.subheader("Result:")
        st.write(f"**{label}** (Confidence: `{confidence * 100:.2f}%`)")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using TensorFlow + Streamlit")
