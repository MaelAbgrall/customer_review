# dependencies
import sklearn
import numpy

# project
import utils.progressBar as progressBar
import utils.textPreprocessing as textPreprocessing


# source: https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/part-1-for-beginners-bag-of-words



def shuffle_set(array):
    """shuffle a numpy array

    Arguments:
        array {numpy.array} -- array to shuffle

    Returns:
        [type] -- [description]
    """
    array = sklearn.utils.shuffle(array)
    return array


def three_split(array, percentage, debug=False):
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
        if(debug): progressBar.progressBar(position, size)
        # train data
        if(position < split):
            train_list.append(array[position])
        # validation data
        if(position >= split):
            validation_list.append(array[position])

    train_array = numpy.array(train_list)
    validation_array = numpy.array(validation_list)

    return train_array, validation_array


def clean_array(array, clean_pattern={"lowercasing":True, "stopword":False, "stemming":False}, debug=False):
    """clean an array
    
    Arguments:
        array {numpy.array} -- numpy array of type (text, label)
    
    Keyword Arguments:
        clean_pattern {dict} -- preprocessing techniques to use. See :func:`~utils.dataUtils.clean_text` for dict parameters (default: {{"lowercasing":True, "stopword":False, "stemming":False}})
        debug {bool} -- show loading bar (default: {False})
    
    Returns:
        numpy.array -- cleaned array
    """

    print("cleaning the text files. Clean pattern: " + str(clean_pattern))

    # loop through the array
    size = array.shape[0]
    for position in range(size):
        # visual feedback
        if(debug): progressBar.progressBar(position, size)
        # extract from DS and clean
        text = array[position, 0]
        text = clean_text(text, clean_pattern)
        # put back the text in the DS
        array[position, 0] = text
    # end of for loop
    if(debug): print("\n")
    return array


def clean_text(text, clean_pattern={"lowercasing":True, "stopword":False, "stemming":False}):
    """clean a string using clean_pattern parameters. this function will always call :func:`~utils.textPreprocessing.noise_removal`
    
    Arguments:
        text {string} -- a text to clean
    
    Keyword Arguments:
        clean_pattern {dict} -- the preprocessing techniques to use. "lowercasing":"F"/"T"/"intelligent" use lowercasing or intelligent lowercasing. See :func:`~utils.textPreprocessing.lowercasing` for more details (default: {{"lowercasing":True, "stopword":False, "stemming":False}})
    
    Returns:
        string -- cleaned text
    """

    # denoising + split in list of string
    word_list = textPreprocessing.noise_removal(text)

    # lowercase
    if(clean_pattern["lowercasing"] == "T"):
        word_list = textPreprocessing.lowercasing(word_list)

    if(clean_pattern["lowercasing"] == "intelligent"):
        word_list = textPreprocessing.lowercasing(word_list, intelligent=True)
    
    # stopwords removal
    if(clean_pattern["stopword"] == True):
        word_list = textPreprocessing.stopWord_removal(word_list)
    
    # stemming
    if(clean_pattern["stemming"] == True):
        word_list = textPreprocessing.stemming(word_list)
        

    # Join the words back into one string separated by space,
    text = " ".join(word_list)

    return text
