
import os
import re
import pickle
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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
    BASE_DIR, "models", "textile_sentiment_lstm.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR, "models", "textile_tokenizer.pkl"
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
# 4. LOAD VADER SENTIMENT ANALYZER
# ==========================================

@st.cache_resource
def load_vader():

    return SentimentIntensityAnalyzer()


analyzer = load_vader()


# ==========================================
# 5. TEXT PREPROCESSING
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
# 6. HYBRID SENTIMENT PREDICTION
# ==========================================

def predict_sentiment(review):

    cleaned_text = clean_review(review)

    if not cleaned_text:
        raise ValueError(
            "No valid English words found in the review."
        )

    # --------------------------------------
    # A. VADER SENTIMENT ANALYSIS
    # --------------------------------------

    vader_result = analyzer.polarity_scores(review)

    vader_score = float(vader_result["compound"])

    if vader_score >= 0.05:
        vader_prediction = "Positive"

    elif vader_score <= -0.05:
        vader_prediction = "Negative"

    else:
        vader_prediction = "Neutral"

    # --------------------------------------
    # B. LSTM SENTIMENT PREDICTION
    # --------------------------------------

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    lstm_score = float(
        model.predict(padded, verbose=0)[0][0]
    )

    # Assumes the LSTM was trained with:
    # 0 = Negative, 1 = Positive

    if lstm_score >= 0.60:
        lstm_prediction = "Positive"

    elif lstm_score < 0.40:
        lstm_prediction = "Negative"

    else:
        lstm_prediction = "Neutral / Uncertain"

    # --------------------------------------
    # C. COMBINE BOTH PREDICTIONS
    # --------------------------------------

    # VADER is used for the final prediction.
    # LSTM provides an additional model prediction.

    final_prediction = vader_prediction

    return (
        final_prediction,
        vader_score,
        lstm_score,
        vader_prediction,
        lstm_prediction,
        cleaned_text
    )


# ==========================================
# 7. STREAMLIT APPLICATION UI
# ==========================================

st.title("👗 Textile Customer Review Sentiment Analysis")

st.write(
    """
    This application uses Natural Language Processing,
    VADER Sentiment Analysis and a trained LSTM Deep Learning
    model to analyze textile customer reviews.
    """
)

st.info(
    "Enter a customer review below to analyze its sentiment."
)


review = st.text_area(
    "Enter Customer Review:",
    placeholder="Example: This dress is beautiful and comfortable.",
    height=130
)


# ==========================================
# 8. ANALYZE SENTIMENT BUTTON
# ==========================================

if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():

        st.warning("Please enter a customer review.")

    else:

        try:

            (
                prediction,
                vader_score,
                lstm_score,
                vader_prediction,
                lstm_prediction,
                cleaned_text
            ) = predict_sentiment(review)

            st.subheader("📊 Sentiment Prediction Result")

            if prediction == "Positive":

                st.success("Positive Review 😊")

            elif prediction == "Negative":

                st.error("Negative Review 😞")

            else:

                st.warning("Neutral Review 😐")

            # Final prediction
            st.markdown("### Final Sentiment")
            st.write(f"**{prediction}**")

            # VADER result
            st.markdown("### VADER Analysis")

            st.write(
                f"VADER Prediction: **{vader_prediction}**"
            )

            st.metric(
                "VADER Compound Score",
                f"{vader_score:.4f}"
            )

            # LSTM result
            st.markdown("### LSTM Deep Learning Analysis")

            st.write(
                f"LSTM Prediction: **{lstm_prediction}**"
            )

            st.metric(
                "LSTM Model Output Score",
                f"{lstm_score:.4f}"
            )

            # Cleaned text
            st.markdown("### Cleaned Review")
            st.write(cleaned_text)

            st.caption(
                "Note: The final sentiment currently follows "
                "VADER. LSTM prediction is displayed separately."
            )

        except Exception as e:

            st.error("Prediction failed.")
            st.exception(e)


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "Developed using Python, NLP, VADER, "
    "TensorFlow, Keras and LSTM Deep Learning."
)
