import re

from src.preprocessing import preprocess_review


# Define aspect keywords
ASPECT_KEYWORDS = {
    'Fit': [
        'fit', 'fitting', 'fits', 'loose',
        'tight', 'flattering', 'shape'
    ],

    'Size': [
        'size', 'sizing', 'small', 'large',
        'bigger', 'smaller', 'length'
    ],

    'Material': [
        'fabric', 'material', 'cotton',
        'silk', 'texture', 'thick', 'thin'
    ],

    'Comfort': [
        'comfortable', 'comfort', 'soft',
        'itchy', 'scratchy', 'wearable'
    ],

    'Design': [
        'design', 'style', 'pattern',
        'print', 'beautiful', 'cute', 'gorgeous'
    ],

    'Colour': [
        'color', 'colour', 'shade',
        'bright', 'dark', 'pale'
    ],

    'Quality': [
        'quality', 'durable', 'cheap',
        'poor', 'stitching', 'damage'
    ],

    'Price': [
        'price', 'expensive', 'cheap',
        'worth', 'value', 'cost'
    ]
}


def analyze_aspects(review_text, title=""):

    # Preprocess review
    processed_text = preprocess_review(review_text, title)

    # Store detected aspects
    detected_aspects = []

    # Check each aspect
    for aspect, keywords in ASPECT_KEYWORDS.items():

        pattern = r'\b(' + '|'.join(
            re.escape(word) for word in keywords
        ) + r')\b'

        if re.search(pattern, processed_text):
            detected_aspects.append(aspect)

    return detected_aspects