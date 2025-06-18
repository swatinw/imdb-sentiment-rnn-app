# imdb-sentiment-rnn-app
🎬 A Streamlit web app that uses a Simple RNN model to perform sentiment analysis on IMDB movie reviews. Enter a review and instantly classify it as positive or negative.

## 🔍 About
This project is an interactive Streamlit app that performs binary sentiment classification on IMDB movie reviews using a trained Simple RNN model in TensorFlow/Keras.

## 🚀 Features
- Pre-trained RNN model for text classification
- Real-time sentiment prediction via Streamlit
- Handles out-of-vocabulary words and padded sequences

## 🧠 Model
- Architecture: Embedding → SimpleRNN → Dense (sigmoid)
- Dataset: IMDB (Keras built-in)
- Trained on 25,000 movie reviews

## 📦 Tech Stack
- Python, TensorFlow/Keras
- Streamlit
- IMDB Dataset (Keras)

## 🖥️ Run Locally
```bash
streamlit run imdb_sentiment_app.py
