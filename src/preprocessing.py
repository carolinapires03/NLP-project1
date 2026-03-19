import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words_en = set(stopwords.words("english"))
stop_words_es = set(stopwords.words("spanish"))
stop_words_ca = ["a","abans","ací","ah","així","això","al","aleshores","algun","alguna","algunes","alguns","alhora","allà","allí","allò","als","altra","altre","altres","amb","ambdues","ambdós","anar","ans","apa","aquell","aquella","aquelles","aquells","aquest","aquesta","aquestes","aquests","aquí","baix","bastant","bé","cada","cadascuna","cadascunes","cadascuns","cadascú","com","consegueixo","conseguim","conseguir","consigueix","consigueixen","consigueixes","contra","d'un","d'una","d'unes","d'uns","dalt","de","del","dels","des","des de","després","dins","dintre","donat","doncs","durant","e","eh","el","elles","ells","els","em","en","encara","ens","entre","era","erem","eren","eres","es","esta","estan","estat","estava","estaven","estem","esteu","estic","està","estàvem","estàveu","et","etc","ets","fa","faig","fan","fas","fem","fer","feu","fi","fins","fora","gairebé","ha","han","has","haver","havia","he","hem","heu","hi","ho","i","igual","iguals","inclòs","ja","jo","l'hi","la","les","li","li'n","llarg","llavors","m'he","ma","mal","malgrat","mateix","mateixa","mateixes","mateixos","me","mentre","meu","meus","meva","meves","mode","molt","molta","moltes","molts","mon","mons","més","n'he","n'hi","ne","ni","no","nogensmenys","només","nosaltres","nostra","nostre","nostres","o","oh","oi","on","pas","pel","pels","per","per que","perquè","però","poc","poca","pocs","podem","poden","poder","podeu","poques","potser","primer","propi","puc","qual","quals","quan","quant","que","quelcom","qui","quin","quina","quines","quins","què","s'ha","s'han","sa","sabem","saben","saber","sabeu","sap","saps","semblant","semblants","sense","ser","ses","seu","seus","seva","seves","si","sobre","sobretot","soc","solament","sols","som","son","sons","sota","sou","sóc","són","t'ha","t'han","t'he","ta","tal","també","tampoc","tan","tant","tanta","tantes","te","tene","tenim","tenir","teniu","teu","teus","teva","teves","tinc","ton","tons","tot","tota","totes","tots","un","una","unes","uns","us","va","vaig","vam","van","vas","veu","vosaltres","vostra","vostre","vostres","érem","éreu","és","éssent","últim","ús"]

lemmatizer = WordNetLemmatizer()

#elimina a pontuação
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

#não elimina a pontuação
'''def clean_text_light(text):
    """
    Light text cleaning:
    - lowercase
    - remove URLs
    - remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text'''


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
        stop_words = stop_words_ca

    return [word for word in tokens if word not in stop_words]


def lemmatize_tokens(tokens, lang):
    """
    Lemmatize tokens.
    Applied only to English because WordNetLemmatizer is English-based.
    """
    if lang == "en":
        return [lemmatizer.lemmatize(word) for word in tokens]
    else:
        return tokens


def preprocess_text(text, lang):
    """
    Full preprocessing pipeline
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, lang)
    tokens = lemmatize_tokens(tokens, lang)

    return " ".join(tokens)

'''def preprocess_text(text, lang):
    """
    Light preprocessing pipeline
    """
    text = clean_text_light(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, lang)
    tokens = lemmatize_tokens(tokens, lang)
    return " ".join(tokens)'''
