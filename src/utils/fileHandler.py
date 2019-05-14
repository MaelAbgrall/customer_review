# python modules
import os

# dependencies
import numpy

# project
import utils.progressBar as progressBar


def import_dataset(path, set_type):
    """import the imdb movie review dataset

    Arguments:
        path {string} -- path to the dataset
        type {string} -- type of the dataset to import ("test" or "train")

    Returns:
        dataset -- numpy array of shape (nb_sample, 2) type: [text, label]
    """
    dataset = []

    for root, _, files in os.walk(path):

        # train set
        if "/train/pos" in root or "/train/neg" in root:
            if set_type == "train":
                print("\n" + root)
                dataset.extend(open_files(root, files))

        # test set
        if "/test/pos" in root or "/test/neg" in root:
            if set_type == "test":
                print("\n" + root)
                dataset.extend(open_files(root, files))

    #data_type = numpy.dtype('string,int')

    dataset = numpy.array(dataset)
    return dataset


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


def convert_to_npy(path):
    """convert the IMDB dataset to a numpy binary file (faster load)

    Arguments:
        path {string} -- path the the root folder
    """
    dataset = import_dataset("../dataset", "train")
    save_np_array("../dataset/train", dataset)
    dataset = import_dataset("../dataset", "test")
    save_np_array("../dataset/test", dataset)


def save_np_array(path, array):
    """save the numpy array to a binary file

    Arguments:
        path {string} -- path to the file to save
        array {numpy.array} -- numpy array
    """
    numpy.save(path, array)


def load_np_array(path):
    """load a numpy array from a npy file

    Arguments:
        path {string} -- path the the file.npy

    Returns:
        [type] -- [description]
    """
    array = numpy.load(path)
    return array
