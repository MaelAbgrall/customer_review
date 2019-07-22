""" This script will convert the IMDB dataset into a numpy file (for faster load)
        and clean it using all possible pattern combination
"""
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

#############
#  Imports  #
#############
#python integrated
import pathlib

# project files
import utils.fileHandler as fileHandler
import utils.dataUtils as dataUtils

##########
# Static #
##########
FOLDER = "../dataset/cleaned_sets"
pathlib.Path(FOLDER).mkdir(parents=True, exist_ok=True)

TRAIN_SET = fileHandler.import_dataset("../dataset", "train")
TEST_SET = fileHandler.import_dataset("../dataset", "test")


#######################
# preprocessing types #
#######################
clean_patterns = []
# F F
clean_patterns.append({"lowercasing":"T", "stopword":False, "stemming":False})
clean_patterns.append({"lowercasing":"intelligent", "stopword":False, "stemming":False})
clean_patterns.append({"lowercasing":"F", "stopword":False, "stemming":False})

# T F
clean_patterns.append({"lowercasing":"T", "stopword":True, "stemming":False})
clean_patterns.append({"lowercasing":"intelligent", "stopword":True, "stemming":False})
clean_patterns.append({"lowercasing":"F", "stopword":True, "stemming":False})

# F T
clean_patterns.append({"lowercasing":"T", "stopword":False, "stemming":True})
clean_patterns.append({"lowercasing":"intelligent", "stopword":False, "stemming":True})
clean_patterns.append({"lowercasing":"F", "stopword":False, "stemming":True})

# T T
clean_patterns.append({"lowercasing":"T", "stopword":True, "stemming":True})
clean_patterns.append({"lowercasing":"intelligent", "stopword":True, "stemming":True})
clean_patterns.append({"lowercasing":"F", "stopword":True, "stemming":True})


##########
# Script #
##########
for pos in range(len(clean_patterns)):
    name = clean_patterns[pos]["lowercasing"] + "_" + str(clean_patterns[pos]["stopword"]) + "_" + str(clean_patterns[pos]["stemming"])
    print(name)

    # cleaning
    train_cleanset = dataUtils.clean_array(TRAIN_SET, clean_patterns[pos], debug=True)
    test_cleanset = dataUtils.clean_array(TEST_SET, clean_patterns[pos], debug=True)
    
    # saving
    location = FOLDER + "/" + name
    print("saving to: " + location)
    pathlib.Path(location).mkdir(parents=True, exist_ok=True)
    """ too huge files
    fileHandler.save_np_array(location + "/train", train_cleanset)
    fileHandler.save_np_array(location + "/test", test_cleanset)"""
    fileHandler.save_toCSV(location + "/train.csv", train_cleanset)
    fileHandler.save_toCSV(location + "/test.csv", test_cleanset)