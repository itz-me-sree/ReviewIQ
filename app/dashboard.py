import sys
from pathlib import Path

import streamlit as st
import pandas as pd


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ReviewIQ | Customer Review Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #F5F7FB;
    color: #1E293B;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Main headings */
h1, h2, h3 {
    color: #172554 !important;
    font-weight: 700 !important;
}

/* Normal text */
p, label, span, li {
    color: #334155;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    color: #1D4ED8 !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

div[data-testid="stMetricValue"] {
    color: #1D4ED8 !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background-color: #2563EB;
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.3rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white !important;
    border: none;
}

/* Text area */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
}

/* Input fields */
.stTextInput input {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF;
}

/* Dividers */
hr {
    border-color: #E2E8F0;
}

/* Info, success and warning boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Navigation radio buttons */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 8px 10px;
    border-radius: 7px;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border-radius: 10px;
}

/* Small description text */
.subtitle {
    color: #64748B;
    font-size: 15px;
}

/* Section spacing */
.section-gap {
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PROJECT PATHS AND IMPORTS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

from src.prediction import analyze_review

DATA_PATH = BASE_DIR / "data" / "processed" / "topic_analyzed_reviews.csv"


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()

except FileNotFoundError:
    st.error("Dataset not found. Please check the processed data folder.")
    st.stop()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📊 ReviewIQ")

st.sidebar.caption("Intelligent Customer Review Analytics")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Sentiment Analytics",
        "Aspect & Topic Analysis",
        "Review Analyzer"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### About ReviewIQ")

st.sidebar.caption(
    "An NLP-powered customer review analytics system "
    "that extracts sentiment, aspects and topics from reviews."
)

st.sidebar.divider()

st.sidebar.caption("Built using Python, Machine Learning and NLP")


# --------------------------------------------------
# COMMON HEADER
# --------------------------------------------------

st.title("ReviewIQ")

st.markdown(
    '<p class="subtitle">Intelligent Customer Review Analytics System</p>',
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Dataset Overview")

    st.markdown(
        '<p class="subtitle">A quick summary of customer reviews and their sentiment.</p>',
        unsafe_allow_html=True
    )

    total_reviews = len(df)

    positive_reviews = (df["Sentiment"] == "Positive").sum()
    negative_reviews = (df["Sentiment"] == "Negative").sum()
    neutral_reviews = (df["Sentiment"] == "Neutral").sum()

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Reviews", f"{total_reviews:,}")

    with col2:
        st.metric("Positive Reviews", f"{positive_reviews:,}")

    with col3:
        st.metric("Negative Reviews", f"{negative_reviews:,}")

    with col4:
        st.metric("Neutral Reviews", f"{neutral_reviews:,}")

    st.divider()

    # Dataset information
    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dataset Details**")

        st.write(f"Total records: {total_reviews:,}")
        st.write(f"Total columns: {df.shape[1]}")
        st.write("Dataset: Women's E-Commerce Clothing Reviews")

    with col2:
        st.markdown("**Sentiment Distribution**")

        sentiment_counts = df["Sentiment"].value_counts()

        st.bar_chart(sentiment_counts)

    st.divider()

    # Department distribution
    st.subheader("Reviews by Department")

    department_counts = df["Department Name"].value_counts()

    department_counts = department_counts.drop(
        index="Unknown",
        errors="ignore"
    )

    st.bar_chart(department_counts)


# ==================================================
# PAGE 2: SENTIMENT ANALYTICS
# ==================================================

elif page == "Sentiment Analytics":

    st.header("Customer Sentiment Insights")

    st.markdown(
        '<p class="subtitle">Understand how customers feel about the products.</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Sentiment distribution
    with col1:

        st.subheader("Overall Sentiment Distribution")

        sentiment_counts = df["Sentiment"].value_counts()

        st.bar_chart(sentiment_counts)

    # Department-wise sentiment
    with col2:

        st.subheader("Department-wise Sentiment")

        department_sentiment = pd.crosstab(
            df["Department Name"],
            df["Sentiment"],
            normalize="index"
        ) * 100

        department_sentiment = department_sentiment.drop(
            index="Unknown",
            errors="ignore"
        )

        st.bar_chart(department_sentiment)

    st.divider()

    # Rating distribution
    st.subheader("Rating Distribution")

    rating_counts = df["Rating"].value_counts().sort_index()

    st.bar_chart(rating_counts)

    st.divider()

    # Sentiment percentages
    st.subheader("Sentiment Percentages")

    sentiment_percentages = (
        df["Sentiment"]
        .value_counts(normalize=True) * 100
    ).round(2)

    percentage_df = sentiment_percentages.reset_index()

    percentage_df.columns = ["Sentiment", "Percentage"]

    st.dataframe(
        percentage_df,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# PAGE 3: ASPECT & TOPIC ANALYSIS
# ==================================================

elif page == "Aspect & Topic Analysis":

    st.header("Customer Aspect Insights")

    st.markdown(
        '<p class="subtitle">Explore the product features customers mention in their reviews.</p>',
        unsafe_allow_html=True
    )

    aspect_columns = [
        "Fit",
        "Size",
        "Material",
        "Comfort",
        "Design",
        "Colour",
        "Quality",
        "Price"
    ]

    col1, col2 = st.columns(2)

    # Aspect frequency
    with col1:

        st.subheader("Frequently Mentioned Aspects")

        aspect_counts = (
            df[aspect_columns]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(aspect_counts)

    # Aspect sentiment
    with col2:

        st.subheader("Aspect-wise Sentiment")

        aspect_sentiment_data = {}

        for aspect in aspect_columns:

            aspect_reviews = df[df[aspect] == True]

            percentages = (
                aspect_reviews["Sentiment"]
                .value_counts(normalize=True) * 100
            )

            aspect_sentiment_data[aspect] = percentages

        aspect_sentiment_df = (
            pd.DataFrame(aspect_sentiment_data)
            .T
            .fillna(0)
            .round(2)
        )

        st.bar_chart(aspect_sentiment_df)

    st.divider()

    # Aspect details
    st.subheader("Aspect Sentiment Details")

    aspect_details = []

    for aspect in aspect_columns:

        aspect_reviews = df[df[aspect] == True]

        sentiment_percentages = (
            aspect_reviews["Sentiment"]
            .value_counts(normalize=True) * 100
        )

        aspect_details.append({
            "Aspect": aspect,
            "Total Reviews": len(aspect_reviews),
            "Positive (%)": round(
                sentiment_percentages.get("Positive", 0), 2
            ),
            "Neutral (%)": round(
                sentiment_percentages.get("Neutral", 0), 2
            ),
            "Negative (%)": round(
                sentiment_percentages.get("Negative", 0), 2
            )
        })

    aspect_details_df = pd.DataFrame(aspect_details)

    st.dataframe(
        aspect_details_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Topic analysis
    st.header("Customer Review Topic Insights")

    st.markdown(
        '<p class="subtitle">Discover the major themes discussed in customer reviews.</p>',
        unsafe_allow_html=True
    )

    topic_counts = df["Topic_Name"].value_counts()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Topic Distribution")

        st.bar_chart(topic_counts)

    with col2:

        st.subheader("Topic Percentages")

        topic_percentages = (
            df["Topic_Name"]
            .value_counts(normalize=True) * 100
        ).round(2)

        topic_percentage_df = topic_percentages.reset_index()

        topic_percentage_df.columns = ["Topic", "Percentage"]

        st.dataframe(
            topic_percentage_df,
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# PAGE 4: REVIEW ANALYZER
# ==================================================

elif page == "Review Analyzer":

    st.header("Customer Review Analyzer")

    st.markdown(
        '<p class="subtitle">Enter a customer review to predict its sentiment, identify aspects and discover its topic.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    # Review input
    st.subheader("Enter Customer Review")

    review_title = st.text_input(
        "Review Title (Optional)",
        placeholder="Example: Beautiful summer dress"
    )

    review_text = st.text_area(
        "Review Text",
        placeholder="Type or paste a customer review here...",
        height=180
    )

    analyze_button = st.button(
        "Analyze Review",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not review_text.strip():

            st.warning("Please enter a review before analyzing.")

        else:

            try:

                with st.spinner("Analyzing customer review..."):

                    results = analyze_review(
                        review_text,
                        review_title
                    )

                st.success("Review analysis completed!")

                st.divider()

                # Sentiment and topic
                st.subheader("Analysis Results")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("**Predicted Sentiment**")

                    sentiment = results["Sentiment"]

                    if sentiment == "Positive":
                        st.success("😊 Positive")

                    elif sentiment == "Negative":
                        st.error("😞 Negative")

                    else:
                        st.info("😐 Neutral")

                with col2:

                    st.markdown("**Dominant Topic**")

                    st.info(results["Dominant Topic"])

                st.divider()

                # Detected aspects
                st.subheader("Detected Product Aspects")

                detected_aspects = results["Detected Aspects"]

                if detected_aspects:

                    aspect_columns = st.columns(4)

                    for index, aspect in enumerate(detected_aspects):

                        with aspect_columns[index % 4]:
                            st.success(f"✓ {aspect}")

                else:

                    st.info("No predefined product aspects were detected.")

                st.divider()

                st.caption(
                    "Note: Predictions are generated using trained "
                    "Machine Learning and NLP models. Sentiment labels "
                    "in the training dataset were derived from product ratings."
                )

            except Exception as e:

                st.error(f"An error occurred during analysis: {e}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748B; padding:10px;">
        ReviewIQ | Intelligent Customer Review Analytics System
        <br>
        Built with Python, Machine Learning and Natural Language Processing
    </div>
    """,
    unsafe_allow_html=True
)