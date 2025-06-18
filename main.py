
# Step 1: Import libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model("simple_rnn_imdb.h5")

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

# Function to preprocess user input
VOCAB_SIZE = 10000

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        index = word_index.get(word, 2) + 3  # shift for special tokens
        if index >= VOCAB_SIZE:
            index = 2  # Treat as OOV
        encoded_review.append(index)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
import streamlit as st

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("📝 Enter a movie review below and let the AI classify it as 📈 Positive or 📉 Negative!")

# User input
user_input = st.text_area("🗣️ Movie Review")

if st.button("🔍 Classify Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)

        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        # Display the result
        st.write(f"🧠 **Sentiment:** `{sentiment}`")
        st.write(f"Prediction Score: {prediction[0][0]:.4f}")
else:
    st.info("💡 Tip: Write a short review above and click the button to classify!")
