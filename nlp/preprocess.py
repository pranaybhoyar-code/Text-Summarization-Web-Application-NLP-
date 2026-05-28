import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize text into words
    words = word_tokenize(text)

    # Load English stopwords
    stop_words = set(stopwords.words("english"))

    # Remove stopwords
    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    # Join cleaned words
    cleaned_text = " ".join(filtered_words)

    return cleaned_text.strip()