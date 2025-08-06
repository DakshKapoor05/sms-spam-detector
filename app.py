import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved tokenizer
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load tokenizer from JSON
with open("tokenizer_config.json", "r") as f:
    tokenizer_json = f.read()

tokenizer = tokenizer_from_json(tokenizer_json)
# Load the trained model
model = tf.keras.models.load_model("sms_spam_model.h5")

# Constants
vocab_size = 5000
max_length = 100

# App title
st.set_page_config(page_title="SMS Spam Detection", layout="centered")
st.title("📱 SMS Spam Detection App")
st.markdown("Enter an SMS message below to check whether it's **spam** or **ham**.")

# Input box
user_input = st.text_area("✉️ Enter SMS message:", height=150)

# Prediction function
def classify_message(message):
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)[0][0]
    return prediction

# Button
if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        pred = classify_message(user_input)
        if pred >= 0.5:
            st.error("🚫 This message is **Spam**.")
        else:
            st.success("✅ This message is **Ham** (Not Spam).")

# Footer
st.markdown("---")

