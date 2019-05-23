# project files
import utils.fileHandler as fileHandler
import utils.dataUtils as dataUtils
import models.WordModels.bagOfWord as bagOfWord
import utils.textPreprocessing as tp

def convert_clean():
    """convert the IMDB dataset to a numpy binary file (faster load)
    """
    # train
    dataset = fileHandler.import_dataset("../dataset", "train")
    dataset = dataUtils.clean_array(dataset, debug=True)
    fileHandler.save_np_array("../dataset/train", dataset)
    
    # test
    dataset = fileHandler.import_dataset("../dataset", "test")
    dataset = dataUtils.clean_array(dataset, debug=True)
    fileHandler.save_np_array("../dataset/test", dataset)

#convert_clean()

trainset = fileHandler.load_np_array("../dataset/train.npy")
trainset = dataUtils.shuffle_set(trainset)
trainset = dataUtils.clean_array(trainset, debug=True)

#train, validation = dataUtils.three_split(trainset, 0.75)

model = bagOfWord.bagOfWord("TFIDF")
model.train(trainset[:, 0])
plop = model.process(trainset[:, 0])
features = model.get_features()
import ipdb; ipdb.set_trace()