import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Basic text cleaning:
    - lowercase
    - remove URLs
    - remove numbers
    - remove punctuation
    - remove extra spaces
    """
    
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    text = text.strip()

    return text


def tokenize(text):
    """
    Tokenize text into words
    """
    
    return word_tokenize(text)


def remove_stopwords(tokens):
    """
    Remove common English stopwords
    """
    
    return [word for word in tokens if word not in stop_words]


def lemmatize_tokens(tokens):
    """
    Lemmatize tokens
    """
    
    return [lemmatizer.lemmatize(word) for word in tokens]


def preprocess_text(text):
    """
    Full preprocessing pipeline
    """
    
    text = clean_text(text)

    tokens = tokenize(text)

    tokens = remove_stopwords(tokens)

    tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)