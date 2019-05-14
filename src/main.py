# project files
import utils.fileHandler as fileHandler
import utils.dataUtils as dataUtils

#fileHandler.convert_to_npy("../dataset")

trainset = fileHandler.load_np_array("../dataset/train.npy")
trainset = dataUtils.shuffle_set(trainset)
train, validation = dataUtils.three_split(trainset, 0.75)
