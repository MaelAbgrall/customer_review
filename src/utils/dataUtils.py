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
        if(debug):
            progressBar.progressBar(position, size)
        # train data
        if(position < split):
            train_list.append(array[position])
        # validation data
        if(position >= split):
            validation_list.append(array[position])

    train_array = numpy.array(train_list)
    validation_array = numpy.array(validation_list)

    return train_array, validation_array


def crossvalidation_split(array, folds, debug=False):
    """take a numpy array and split it in x number of subset for cross validation

    Arguments:
        folds {int} -- number of subsets
        array {numpy.array} -- [description]

    Returns:
        list of subsets
        the list of sets is segmented as following:
        [(train, validation), (train, validation), ...]
    """
    print("too memory intensive, please replace dataUtils.crossvalidation_split by dataUtils.cv_split")
    exit()

    # we split our array in a list of small sets
    small_batch = numpy.array_split(array, folds)
    # each of the element of this list is a array: small_batch[0] is the subset 0 with the X and Y

    crossvalidation_arrays = []
    # for each subset we want to create
    for set_number in range(folds):
        # we will concatenate every batch created earlier and use the batch at [position] as validation array
        """
            example:
            batch a
            batch b
            batch c
            batch d
            subset 1 = b + c + d
            subset 2 = a + c + d
            subset 3 = a + b + d
            subset 4 = a + b + c
        """

        # for faster operation: creation of an empty numpy array
        lines = 0
        for position in range(len(small_batch)):
            if(position != set_number):
                lines += small_batch[position].shape[0]
        # if it's a 2D array
        if(len(small_batch[0].shape) > 1):
            rows = small_batch[0].shape[1]
            train = numpy.zeros((lines, rows))
        # if it's labels
        if(len(small_batch[0].shape) == 1):
            train = numpy.zeros((lines,))

        # populating the arrays
        memory_start = 0
        memory_end = memory_start + small_batch[0].shape[0]
        for position in range(folds):

            if(position == set_number):
                validation = small_batch[position]

            if(position != set_number):
                train[memory_start:memory_end] = small_batch[position]

            # updating the memory range
            if(position+1 < len(small_batch) and position != set_number):
                memory_start = memory_end
                memory_end = memory_start + small_batch[position+1].shape[0]

        # end of "for position in range(folds):"
        crossvalidation_arrays.append((train, validation))
    # end of "for set_number in range(folds):"
    return crossvalidation_arrays


def cv_split(array, folds):
    """take a numpy array and split it in x number of subset for cross validation
    """
    # we split our array in a list of small sets
    small_batches = numpy.array_split(array, folds)

    return small_batches


def create_sets_forCV(small_batch, set_number, folds):
    """create a train and validation set + labels for cross validation
    used in conbination with cv_split instead of crossvalidation_split

    Arguments:
        small_batch {list} -- result of cv_split
        set_number {int} -- the set number
    """
    # for faster operation: creation of an empty numpy array
    lines = 0
    for position in range(len(small_batch)):
        if(position != set_number):
            lines += small_batch[position].shape[0]
    # if it's a 2D array
    if(len(small_batch[0].shape) > 1):
        rows = small_batch[0].shape[1]
        train = numpy.zeros((lines, rows))
    # if it's labels
    if(len(small_batch[0].shape) == 1):
        train = numpy.zeros((lines,))

    # populating the arrays
    memory_start = 0
    memory_end = memory_start + small_batch[0].shape[0]
    for position in range(folds):

        if(position == set_number):
            validation = small_batch[position]

        if(position != set_number):
            train[memory_start:memory_end] = small_batch[position]

        # updating the memory range
        if(position+1 < len(small_batch) and position != set_number):
            memory_start = memory_end
            memory_end = memory_start + small_batch[position+1].shape[0]

    # end of "for position in range(folds):"
    return (train, validation)







def clean_array(array, clean_pattern={"lowercasing": "T", "stopword": False, "stemming":False}, debug=False):
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
        if(debug):
            progressBar.progressBar(position, size)
        # extract from DS and clean
        text = array[position, 0]
        text = clean_text(text, clean_pattern)
        # put back the text in the DS
        array[position, 0] = text
    # end of for loop
    if(debug):
        print("\n")
    return array


def clean_text(text, clean_pattern={"lowercasing": "T", "stopword": False, "stemming":False}):
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


def split_text(array, debug=False):
    """split the string in the array

    Arguments:
        array {numpy.array} -- numpy array of type (texts, label)

    Returns:
        numpy.array -- an array of type (list of words in a text, label)
    """
    new_array = []
    print("splitting text into lists")
    # loop through the array
    size = array.shape[0]
    for position in range(size):
        # visual feedback
        if(debug):
            progressBar.progressBar(position, size)
        # extract from DS and split
        text = array[position, 0]
        split_text = text.split()
        # put back the text in the DS
        new_array.append([split_text, array[position, 1]])
    # end of for loop
    print("\nconverting list to numpy array")
    array = numpy.array(new_array)
    print("split_text done...\n")
    return array
