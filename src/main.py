# project files
import utils.fileHandler as fileHandler
import utils.dataUtils as dataUtils
import models.WordModels.bagOfWord as bagOfWord
import utils.textPreprocessing as tp

import models.WordModels.wordEmbedding as wordEmbedding

# model types:
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
TEST_MODEL = [LogisticRegression(), DecisionTreeClassifier(), GaussianNB()]
TEST_MODEL = ["logreg", "dectree", "gaussNB"]

# preprocessing
# ez to update: just add a string to the correct word model
BAG_OF_WORD = ["TF", "TFIDF"]
WORD_EMBEDDING = ["w2v"]
PREPROCESS_MODEL = BAG_OF_WORD + WORD_EMBEDDING

def create_word_model(model_type):
    """create a word model. BEWARE: the text array should be formated accordingly for the model type
    
    Arguments:
        model_type {[type]} -- type of model
    
    Returns:
        model
    """
    if(model_type in BAG_OF_WORD):
        model = bagOfWord.bagOfWord(model_type)

    if(model_type in WORD_EMBEDDING):
        model = wordEmbedding.wordEmbedding(model_type)

    return model


# for each dataset -> load
#   for each preprocess -> create word model
#     for each ML model -> train and evaluate

# load the dataset
#train_array = fileHandler.load_np_array("../dataset/train.npy")
#train_array = dataUtils.shuffle_set(train_array)
#train_array = dataUtils.clean_array(train_array, debug=True)
#train, validation = dataUtils.three_split(train_array, 0.75)

train_array = fileHandler.import_dataset("../dataset", "test")

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

trainDataVecs = model.process(train_array[:, 0])
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