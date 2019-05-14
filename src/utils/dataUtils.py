# dependencies
import sklearn
import numpy

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
    size = array.shape[0]
    print("size" + str(size))
    split = int(size * percentage)
    print("split" + str(split))

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
