import joblib
from pathlib import Path

from src.preprocessing import preprocess_review


# Find project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATH = BASE_DIR / "models" / "topic_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "topic_vectorizer.pkl"
TOPIC_NAMES_PATH = BASE_DIR / "models" / "topic_names.pkl"


# Load saved NMF model, vectorizer and topic names
topic_model = joblib.load(MODEL_PATH)
topic_vectorizer = joblib.load(VECTORIZER_PATH)
topic_names = joblib.load(TOPIC_NAMES_PATH)


def predict_topic(review_text, title=""):

    # Step 1: Preprocess review
    processed_text = preprocess_review(review_text, title)

    if not processed_text:
        raise ValueError("Please enter a valid review.")

    # Step 2: Convert review into TF-IDF features
    review_vector = topic_vectorizer.transform([processed_text])

    # Step 3: Transform review into topic weights
    topic_weights = topic_model.transform(review_vector)

    # Step 4: Find dominant topic
    dominant_topic = topic_weights.argmax(axis=1)[0] + 1

    # Step 5: Get topic name
    topic_name = topic_names[dominant_topic]

    return topic_name