import os
import re
import pickle
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Textile Sentiment Analysis",
    page_icon="👗",
    layout="centered"
)


# ==========================================
# 2. FILE PATH CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "textile_sentiment_lstm.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "textile_tokenizer.pkl"
)

MAX_LEN = 100


# ==========================================
# 3. LOAD LSTM MODEL AND TOKENIZER
# ==========================================

@st.cache_resource
def load_resources():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"LSTM model not found: {MODEL_PATH}"
        )

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer file not found: {TOKENIZER_PATH}"
        )

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


try:
    model, tokenizer = load_resources()

except Exception as e:
    st.error("Unable to load the LSTM model or tokenizer.")
    st.exception(e)
    st.stop()


# ==========================================
# 4. TEXT PREPROCESSING
# ==========================================

def clean_review(text):

    text = text.lower()

    # Expand common contractions
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)

    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'d", " would", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 5. SENTIMENT PREDICTION
# ==========================================

def predict_sentiment(review):

    cleaned_text = clean_review(review)

    if not cleaned_text:
        raise ValueError(
            "No valid English words found in the review."
        )

    # Convert text into numerical sequences
    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # LSTM prediction
    score = float(
        model.predict(padded, verbose=0)[0][0]
    )

    # Classification thresholds
    if score >= 0.60:
        prediction = "Positive"

    elif score < 0.40:
        prediction = "Negative"

    else:
        prediction = "Neutral / Uncertain"

    return prediction, score, cleaned_text


# ==========================================
# 6. STREAMLIT APPLICATION UI
# ==========================================

st.title("👗 Textile Customer Review Sentiment Analysis")

st.write(
    """
    This application uses Natural Language Processing
    and a trained LSTM Deep Learning model to analyze
    textile customer reviews.
    """
)

st.info("Enter a customer review below to analyze its sentiment.")


# Customer review input
review = st.text_area(
    "Enter Customer Review:",
    placeholder="Example: This dress is beautiful and comfortable.",
    height=130
)


# ==========================================
# 7. ANALYZE SENTIMENT BUTTON
# ==========================================

if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():

        st.warning("Please enter a customer review.")

    else:

        try:

            prediction, score, cleaned_text = predict_sentiment(
                review
            )

            st.subheader("📊 Sentiment Prediction Result")

            if prediction == "Positive":

                st.success("Positive Review 😊")

            elif prediction == "Negative":

                st.error("Negative Review 😞")

            else:

                st.warning("Neutral / Uncertain 😐")

            st.metric(
                label="Model Output Score",
                value=f"{score:.4f}"
            )

            st.write("**Cleaned Review:**")
            st.write(cleaned_text)

            st.progress(
                max(0.0, min(1.0, score))
            )

        except Exception as e:

            st.error("Prediction failed.")
            st.exception(e)


# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Developed using Python, NLP, TensorFlow, "
    "Keras and LSTM Deep Learning."
)
