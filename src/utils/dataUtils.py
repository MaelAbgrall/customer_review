# python integrated
import re

# dependencies
import sklearn
import numpy
from bs4 import BeautifulSoup
import nltk

# project
import utils.progressBar as progressBar


def shuffle_set(array):
    """shuffle a numpy array

    Arguments:
        array {numpy.array} -- array to shuffle

    Returns:
        [type] -- [description]
    """
    array = sklearn.utils.shuffle(array)
    return array


def three_split(array, percentage):
    """separate the dataset in two subset (train and validation) using the percentage

    Arguments:
        array {numpy.array} -- full dataset
        percentage {float} -- percentage of train sample (ex: 0.75)

    Returns:
        train_array, validation_array -- numpy arrays of train and validation data (text, label)
    """
    size = array.shape[0]
    split = int(size * percentage)

    train_list = []
    validation_list = []
    for position in range(size):
        # visual feedback
        progressBar.progressBar(position, size)
        # train data
        if(position < split):
            train_list.append(array[position])
        # validation data
        if(position > split):
            validation_list.append(array[position])

    train_array = numpy.array(train_list)
    validation_array = numpy.array(validation_list)

    return train_array, validation_array


def clean_array(array, rm_stopwords=True):
    """clean the text of an array
    
    Arguments:
        array {numpy.array} -- numpy array of texts (text, label)
    
    Keyword Arguments:
        rm_stopwords {bool} -- should the clean function remove the stopwords? (default: {True})
    
    Returns:
        numpy.array -- cleaned array
    """
    # download the stopwords library
    nltk.download('stopwords')

    print("cleaning the text files. Remove stopwords: " + str(rm_stopwords))

    # loop through the array
    size = array.shape[0]
    for position in range(size):
        # visual feedback
        progressBar.progressBar(position, size)
        # extract from DS and clean
        text = array[position, 0]
        clean_text(text, rm_stopwords)
        # put back the text in the DS
        array[position, 0] = text
    # end of for loop

    return array


def clean_text(text, rm_stopwords=True):
    """clean a text

    Arguments:
        text {string} -- text to clean
        rm_stopwords {boolean} -- should the clean function remove stopwords?

    Returns:
        text -- cleaned text as a string
    """
    # source: https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/part-1-for-beginners-bag-of-words
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove non-letters
    text = re.sub("[^a-zA-Z]", " ", text)

    if rm_stopwords == True:
        # Convert to lower case, split into individual words
        word_list = text.lower().split()

        # In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(nltk.corpus.stopwords.words("english"))

        # Remove stop words
        word_list = [w for w in word_list if not w in stops]

        # Join the words back into one string separated by space,
        text = " ".join(word_list)
    # end of stopwords block

    return text
