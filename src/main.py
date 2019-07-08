# project files
import utils.fileHandler as fileHandler
import utils.dataUtils as dataUtils
import models.WordModels.bagOfWord as bagOfWord
import utils.textPreprocessing as tp

import models.WordModels.wordEmbedding as wordEmbedding

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

#convert_clean(); exit()

# load the dataset
train_array = fileHandler.load_np_array("../dataset/train.npy")
train_array = dataUtils.shuffle_set(train_array)
#train_array = dataUtils.clean_array(train_array, debug=True)
#train, validation = dataUtils.three_split(train_array, 0.75)

# bag of word
"""model = bagOfWord.bagOfWord("TFIDF")
model.train(train_array[:, 0])
plop = model.process(train_array[:, 0])
features = model.get_features()"""

# word embedding
print("creating model")
model = wordEmbedding.wordEmbedding("w2v")
print("done...\n")

train_array = dataUtils.split_text(train_array)

model.train(train_array[:, 0])

trainDataVecs = model.process_average(train_array)


import ipdb; ipdb.set_trace()


"""
# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print("Fitting a random forest to labeled training data...")
forest = forest.fit( trainDataVecs, train["sentiment"] )
"""
# Test & extract results 
#result = forest.predict( testDataVecs )

# Write the test results 
"""output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )"""



import ipdb; ipdb.set_trace()