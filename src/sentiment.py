import joblib
from pathlib import Path

from src.preprocessing import preprocess_review


# Find project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# Load saved model and vectorizer
sentiment_model = joblib.load(MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(review_text, title=""):

    # Step 1: Preprocess the review
    processed_text = preprocess_review(review_text, title)

    if not processed_text:
        raise ValueError("Please enter a valid review.")

    # Step 2: Convert review into TF-IDF features
    review_vector = tfidf_vectorizer.transform([processed_text])

    # Step 3: Predict sentiment
    prediction = sentiment_model.predict(review_vector)[0]

    return prediction