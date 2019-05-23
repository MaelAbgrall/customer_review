# python integrated

# dependencies
import sklearn

# project files 
import utils.fileHandler as fileHandler


class bagOfWord():
    # type of model, can be term frequency (TF), TF-IDF
    #model_type = "TF"
    # model
    
    # type: term frequency or TFIDF
    def __init__(self, model_type, path=None):
        self.model_type = model_type
        
        # create a TF model
        if(path is None and model_type == "TF"):
            self.model = sklearn.feature_extraction.text.CountVectorizer()

        # create a TFIDF model
        if(path is None and model_type == "TFIDF"):
            self.model = sklearn.feature_extraction.text.TfidfVectorizer()

        # load a model from pickle file
        if(path):
            self.model = fileHandler.load_pickle(path)

        if(path == None and model_type != "TF"):
            if(model_type != "TFIDF"):
                print("please enter a correct model type: TF or TFIDF. Or provide a path to a pickle file containing a BoW model")
                exit()


    def train(self, texts_array):
        self.model.fit(texts_array)
        

    def process(self, texts_array):
        """transform an array of texts using the model specified (TF or TFIDF)
        
        Arguments:
            texts_array {numpy.array} -- array of texts
        
        Returns:
            numpy.array -- result array
        """
        texts_array = self.model.transform(texts_array)
        texts_array = texts_array.toarray()
        return texts_array

    def get_features(self):
        return self.model.get_feature_names()
    
    # use pickle
    def save(self, path):
        fileHandler.save_pickle(self.model, path)