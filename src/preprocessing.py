import re
import html
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources if unavailable
nltk_resources = [
    'punkt',
    'punkt_tab',
    'stopwords',
    'wordnet',
    'omw-1.4'
]

for resource in nltk_resources:
    try:
        if resource in ['punkt', 'punkt_tab']:
            nltk.data.find(f'tokenizers/{resource}')
        else:
            nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))

# Preserve negation words
stop_words = stop_words - {'no', 'not', 'nor'}

lemmatizer = WordNetLemmatizer()


# Expand common contractions
contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "cannot",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "won't": "will not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not"
}

contraction_pattern = re.compile(
    r"\b(" + "|".join(
        re.escape(word)
        for word in sorted(contractions, key=len, reverse=True)
    ) + r")\b",
    flags=re.IGNORECASE
)


def expand_contractions(text):

    return contraction_pattern.sub(
        lambda match: contractions[match.group().lower()],
        text
    )


def preprocess_review(review_text, title=""):

    # Handle missing input
    if review_text is None:
        review_text = ""

    if title is None:
        title = ""

    # Combine title and review text
    text = str(title) + " " + str(review_text)

    # Decode HTML entities
    text = html.unescape(text)

    # Expand contractions
    text = expand_contractions(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z']", ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal and lemmatization
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word.isalpha()
    ]

    # Convert tokens back into a single string
    processed_text = ' '.join(filtered_tokens)

    return processed_text