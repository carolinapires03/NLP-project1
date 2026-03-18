import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words_en = set(stopwords.words("english"))
stop_words_es = set(stopwords.words("spanish"))

#lemmatizer = WordNetLemmatizer()

'''
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

    return text'''

def clean_text_light(text):
    """
    Light text cleaning:
    - lowercase
    - remove URLs
    - remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """
    Tokenize text into words
    """
    return word_tokenize(text)


def remove_stopwords(tokens, lang):
    """
    Remove stopwords according to language
    """
    if lang == "en":
        stop_words = stop_words_en
    elif lang == "es":
        stop_words = stop_words_es
    else:
        stop_words = set()   #Catalan: no stopword list for now

    return [word for word in tokens if word not in stop_words]

'''
def lemmatize_tokens(tokens, lang):
    """
    Lemmatize tokens.
    Applied only to English because WordNetLemmatizer is English-based.
    """
    if lang == "en":
        return [lemmatizer.lemmatize(word) for word in tokens]
    else:
        return tokens'''


'''def preprocess_text(text, lang):
    """
    Full preprocessing pipeline
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, lang)
    tokens = lemmatize_tokens(tokens, lang)

    return " ".join(tokens)'''

def preprocess_text(text, lang):
    """
    Light preprocessing pipeline
    """
    text = clean_text_light(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, lang)
    return " ".join(tokens)

