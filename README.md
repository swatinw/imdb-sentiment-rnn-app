
# 🎬 IMDB Sentiment Analysis App with RNN

A real-time sentiment classifier for movie reviews using a Simple RNN model, deployed via Streamlit.

---

## 🔍 About

This interactive app performs **binary sentiment classification** (Positive/Negative) on IMDB movie reviews using a **pre-trained Simple RNN model** built in TensorFlow/Keras.

Just enter a review and get an instant sentiment prediction!

---

## 🚀 Features

- 🤖 Pre-trained RNN model for movie review sentiment
- ⚡ Real-time prediction powered by Streamlit
- 🧠 Handles out-of-vocabulary words and padded sequences
- 📱 Clean UI for interactive testing

---

## 🧠 Model Architecture

- **Embedding Layer**: Converts word indices into dense vectors  
- **SimpleRNN Layer**: Captures sequential dependencies  
- **Dense Layer**: Sigmoid activation for binary classification  

**Dataset**: IMDB (25,000 movie reviews)  
**Framework**: TensorFlow/Keras

---

## 📦 Tech Stack

| Component         | Tool/Library              |
|------------------|---------------------------|
| Programming Lang | Python                    |
| ML Framework     | TensorFlow / Keras        |
| Frontend         | Streamlit                 |
| Dataset          | IMDB (via `keras.datasets`) |

---

## 🖥️ How to Run Locally

1. Clone this repo:
```bash
git clone https://github.com/your-username/imdb-sentiment-rnn-app.git
cd imdb-sentiment-rnn-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run imdb_sentiment_app.py
```

---

## 🌐 Live Demo

👉 [Click here to try the live app](https://your-streamlit-app-url)

---

## 🙋‍♀️ Author

**Swati Sharma**  
🔗 [LinkedIn](https://www.linkedin.com/in/swati-sharma-17s50s01/)  
📂 [GitHub](https://github.com/swatinw)

---

📬 _Want to collaborate on NLP or AI projects? Let’s connect!_
