# python modules
import os

# dependencies
import numpy

# project
import utils.progressBar as progressBar


def import_dataset(path):
    """import the imdb movie review dataset

    Arguments:
        path {string} -- path to the dataset

    Returns:
        train_set, test_set -- numpy arrays of shape (nb_sample, 2) type: [text, label]
    """
    train_list = []
    test_list = []

    for root, dir, files in os.walk(path):

        # train set
        if "/train/pos" in root or "/train/neg" in root:
            print("\n" + root)
            train_list.extend(open_files(root, files))

        # test set
        if "/test/pos" in root or "/test/neg" in root:
            print("\n" + root)
            test_list.extend(open_files(root, files))

    #data_type = numpy.dtype('string,int')

    train_set = numpy.array(train_list)
    test_set = numpy.array(test_list)
    return (train_set, test_set)


def open_files(root, files):
    """open the file list output from os.walk()

    Arguments:
        root {string} -- root folder
        files {list} -- file name list

    Returns:
        file_list -- list of tuples (text, label)
    """
    file_list = []

    # tracking progress
    nb_Files = len(files)
    pos = 0

    # saving the label
    if "/pos" in root:
        sentiment = 1
    if "/neg" in root:
        sentiment = 0

    # opening files
    for file_path in files:
        # progress
        progressBar.progressBar(pos, nb_Files)

        with open(root + "/" + file_path) as f:
            text = f.read()
            file_list.append((text, sentiment))

        pos += 1
    return file_list
