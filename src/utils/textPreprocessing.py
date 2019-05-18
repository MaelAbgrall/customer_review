# python integrated
import re

# dependencies
from bs4 import BeautifulSoup
import nltk

nltk.download('wordnet')

#SOURCE: https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html

# download the stopwords library & convert to set (faster than lists)
nltk.download('stopwords')
STOP_WORDS = set(nltk.corpus.stopwords.words("english"))

STEMMER = nltk.stem.PorterStemmer()


def lowercasing(list_of_words, intelligent=False):
    """lowercase a given text. The "intelligent mode" will not lowercase words fully uppercase (ex: STOP and Stop)

    Arguments:
        list_of_words {list} -- list of words in a text

    Keyword Arguments:
        intelligent {bool} -- use the intelligent mode or not (default: {False})

    Returns:
        list -- list of words lowercase
    """
    curated_list = []

    for word in list_of_words:
        # if intelligent mode is on and the word fully uppercase
        if(intelligent and word.isupper()):
            curated_list.append(word)

        # if the intelligent mode is on, and the word is not fully uppercase
        if(intelligent and word.isupper() == False):
            curated_list.append(word.lower())

        if(not intelligent):
            curated_list.append(word.lower())
    # end of loop

    return curated_list


def stemming(list_of_words):
    """use the stemming algorithm on a list of words

    Arguments:
        list_of_words {list} -- list of words from a text

    Returns:
        list -- list of words with stemming applied
    """
    curated_list = []

    for word in list_of_words:
        output = STEMMER.stem(word)

        # preserving uppercase words
        if(word.isupper()):
            curated_list.append(output.upper())

        if(word.isupper() == False):
            curated_list.append(output)
    # end of loop
    return curated_list


def noise_removal(text):
    """remove html tags and non letter characters from a given text and return a list of cleaned words
    
    Arguments:
        text {string} -- text
    
    Returns:
        list -- cleaned text
    """
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove non-letters
    text = re.sub("[^a-zA-Z]", " ", text)

    word_list = text.split()
    return word_list


def stopWord_removal(list_of_words):
    """remove stop words using nltk library
    
    Arguments:
        list_of_words {list} -- list of words
    
    Returns:
        list -- list of words without the stop words
    """
    curated_list = [w for w in list_of_words if not w in STOP_WORDS]
    return curated_list