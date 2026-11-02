
import streamlit as st
import pickle
import re
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="Textile Sentiment Analysis",
    page_icon="👗",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_sentiment_model():
    model = load_model("models/textile_sentiment_lstm.keras")

    with open("models/textile_tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer

model, tokenizer = load_sentiment_model()

# IMPORTANT:
# Replace this function with the exact clean_review()
# function you used while training your model.
def clean_review(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction function
def predict_sentiment(review):
    cleaned_text = clean_review(review)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded = pad_sequences(
        sequence,
        maxlen=100,
        padding="post",
        truncating="post"
    )

    score = float(model.predict(padded, verbose=0)[0][0])

    if score >= 0.5:
        sentiment = "Positive 😊"
        confidence = score * 100
    else:
        sentiment = "Negative 😞"
        confidence = (1 - score) * 100

    return sentiment, confidence, score

# Application UI
st.title("👗 Textile Customer Review Sentiment Analysis")

st.write(
    "Enter a customer review to predict whether "
    "the sentiment is Positive or Negative."
)

review = st.text_area(
    "Enter Customer Review:",
    placeholder="Example: This dress is beautiful and comfortable."
)

if st.button("Analyze Sentiment"):
    if review.strip():

        sentiment, confidence, score = predict_sentiment(review)

        st.subheader("Sentiment Analysis Result")

        if score >= 0.5:
            st.success(sentiment)
        else:
            st.error(sentiment)

        st.write(f"Prediction Score: {score:.4f}")
        st.write(f"Confidence: {confidence:.2f}%")

    else:
        st.warning("Please enter a customer review.")

st.divider()

st.caption(
    "Developed using Python, NLP, TensorFlow, "
    "Keras and Regularized LSTM."
)
