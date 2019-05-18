# project files
import utils.fileHandler as fileHandler
import utils.dataUtils as dataUtils

def convert_clean(path):
    """convert the IMDB dataset to a numpy binary file (faster load)

    Arguments:
        path {string} -- path the the root folder
    """
    dataset = fileHandler.import_dataset("../dataset", "train")
    fileHandler.save_np_array("../dataset/train", dataset)
    dataset = fileHandler.import_dataset("../dataset", "test")
    fileHandler.save_np_array("../dataset/test", dataset)

#fileHandler.convert_to_npy("../dataset")

#trainset = fileHandler.load_np_array("../dataset/train.npy")
#trainset = dataUtils.shuffle_set(trainset)
#train, validation = dataUtils.three_split(trainset, 0.75)

#dataUtils.clean_array(trainset, True)

import ipdb; ipdb.set_trace()