from src.sentiment import predict_sentiment
from src.aspect_analysis import analyze_aspects
from src.topic_modeling import predict_topic


def analyze_review(review_text, title=""):

    # Predict sentiment
    sentiment = predict_sentiment(review_text, title)

    # Detect aspects
    aspects = analyze_aspects(review_text, title)

    # Predict dominant topic
    topic = predict_topic(review_text, title)

    # Combine all results
    results = {
        "Sentiment": sentiment,
        "Detected Aspects": aspects,
        "Dominant Topic": topic
    }

    return results