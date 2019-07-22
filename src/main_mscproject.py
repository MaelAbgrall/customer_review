# python integrated
import ipdb
import os

# libraries
import numpy

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
# those strings are used to create the file name
LEARNING_MODEL = {"logreg":LogisticRegression(solver='lbfgs'), "dectree":DecisionTreeClassifier(), "gaussNB":GaussianNB()}
#cross validation
FOLDS = 10


# preprocessing
# ez to update: just add a string to the correct word model
BAG_OF_WORD = ["TF", "TFIDF"]
WORD_EMBEDDING = ["w2v"]
# those strings are used to create the file name
PREPROCESS_MODEL = BAG_OF_WORD + WORD_EMBEDDING


#######################
# Loading & locations #
#######################
SOURCE_FOLDER = "../dataset/cleaned_sets"

##########
# Script #
##########
"""
for more visibility, the script is split in three sections:

- the first part (not in functions) will load sequentially each datasets present in the source folder.
    and call the function to test each word models

- the second part (def test_each_wordmodel) will create a word model as specified in the PREPROCESS_MODEL constant.
    When the word model has been created and trained, this function will call the third part of the program

- the third part of the program will test every ML and DL model as specified in the LEARNING_MODEL constant,
    and save the result in the source folder

roughly:
for each dataset -> load
    for each preprocess -> create word model
        for each ML model -> train, evaluate and save
"""

def save_result(save_path, result_string, wordmodel_type, learningmodel_type):
    with open(save_path + "/" + wordmodel_type + "_" + learningmodel_type + ".txt", "w+") as file:
        file.write(result_string)

def test_each_learning_model(wordmodel_type, wordmodel, labels, processed_reviews, save_path):
    global LEARNING_MODEL # no idea why it forces me to do that the issue comes from the line: model = LEARNING_MODEL[learningmodel_type]

    # TODO cross validation
    #train_labels, validation_labels = dataUtils.three_split(labels, 0.75)
    #train_array, validation_array = dataUtils.three_split(processed_reviews, 0.75)
    
    print("\t\tCreating cross validation subsets for labels")
    CV_labels = dataUtils.cv_split(labels, folds=FOLDS)
    print("\t\tLABELS done...")
    
    print("\t\tCreating cross validation subsets for reviews")
    CV_reviews = dataUtils.cv_split(processed_reviews, folds=FOLDS)
    print("\t\tdone...")

    for learningmodel_type in LEARNING_MODEL:
        crossvalidation_models = []
        crossvalidation_results = numpy.array([])

        for set_number in range(FOLDS):
            print("\t\tCross validation passage " + str(set_number) + " of " + str(FOLDS))
            
            ####
            # Creating datasets for the fold
            ####
            print("\t\t\tCreating fold")
            train_labels, validation_labels = dataUtils.create_sets_forCV(CV_labels, set_number, FOLDS)
            train_feature, validation_feature = dataUtils.create_sets_forCV(CV_reviews, set_number, FOLDS)
            print("\t\t\tdone...")
            ###
            # training
            ###

            # initialisation
            model = LEARNING_MODEL[learningmodel_type]
            print("\t\t\tTraining learning model")
            model.fit(train_feature, train_labels)
            print("\t\t\tdone...")
            
            print("\t\t\tEvaluating model")
            predictions = model.predict(validation_feature)
            # calculate score : number of correct predictions / total of validation reviews
            score = 0
            for position in range(validation_labels.shape[0]):
                if(validation_labels[position] == predictions[position]):
                    score += 1
            result = score / validation_labels.shape[0]
            
            # saving
            crossvalidation_models.append(model)
            crossvalidation_results = numpy.append(crossvalidation_results, result)
            

        # end of "for set_number in range(folds):"
      
        result_string = "Word model: " + wordmodel_type + "\nLearning model: " + learningmodel_type
        result_string += "\nScores: " + str(crossvalidation_results)
        result_string += "\nMean: " + str(numpy.mean(crossvalidation_results)) + "  Standard deviation: " + str(numpy.std(crossvalidation_results))

        # only strings are saved ATM TODO: save wordmodel, learning model
        save_result(save_path, result_string, wordmodel_type, learningmodel_type)


def test_each_wordmodel(review_array, save_path):
    # for each word model
    for wordmodel_type in PREPROCESS_MODEL:
        
        if(wordmodel_type in BAG_OF_WORD):
            wordmodel = bagOfWord.bagOfWord(wordmodel_type)

        # if we are dealing with word embedding, the dataset need to change
        if(wordmodel_type in WORD_EMBEDDING):
            wordmodel = wordEmbedding.wordEmbedding(wordmodel_type)
            review_array = dataUtils.split_text(review_array)
        
        print("\ttraining " + wordmodel_type)
        wordmodel.train(review_array[:, 0])
        print("\tdone...")
        
        print("\tProcessing reviews using the wordmodel")
        processed_reviews = wordmodel.process(review_array[:, 0])
        print("\tdone...")

        # not dealing with images -> numpy array type is 'object', which is not usable by scikit and keras
        labels = review_array[:, 1]
        labels = labels.astype(numpy.int)
        
        # calling part 3 of the program
        test_each_learning_model(wordmodel_type, wordmodel, labels, processed_reviews, save_path)


for root, _, files in os.walk(SOURCE_FOLDER):
    if(files):
        # loading the dataset
        print("\nloading and shuffling " + str(root))
        review_array = fileHandler.load_fromCSV(root + "/train.csv")
        review_array = dataUtils.shuffle_set(review_array)
        print("done...")
        # call part 2 of the program
        test_each_wordmodel(review_array, root)