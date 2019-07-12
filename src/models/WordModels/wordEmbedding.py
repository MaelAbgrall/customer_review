# python integrated
import multiprocessing

# dependencies
import gensim
import numpy

# project files
import utils.progressBar as progressBar


class wordEmbedding ():
    def __init__(self, model_type, num_features=100, path=None):
        self.model_type = model_type
        self.model = None
        self.num_features = num_features

        # if there is a file to load, use gensim intagrated functions
        if(model_type == "w2v" and path is not None):
            self.model = gensim.models.word2vec.Word2Vec(
                workers=multiprocessing.cpu_count())
            self.model.load(path)

    def train(self, text_array):
        print("training the word embedding model " + self.model_type)
        if(self.model is None):
            self.model = gensim.models.word2vec.Word2Vec(
                text_array, workers=multiprocessing.cpu_count(), size=self.num_features)

        self.model.train(
            text_array, total_examples=text_array.shape[0], epochs=10)
        print("done...\n")

    def get_vector(self, word):
        return self.model[word]

    def similar(self, word):
        return self.model.wv.similar_by_word(word)

    def save(self, path):
        self.model.save(path)

    def average_review(self, review):
        """Function to average all of the word vectors in a given paragraph"""

        # Pre-initialize an empty numpy array (for speed)
        total_vector = numpy.zeros((self.num_features,), dtype="float32")

        # Index2word is a list that contains the names of the words in the model's vocabulary
        # Convert it to a set, for speed
        index2word_set = set(self.model.wv.index2word)

        # for each word in the review, if it is in the model's vocabulary, get its feature vector and add it to the total feature vector
        for word in review:
            if word in index2word_set:
                total_vector = numpy.add(total_vector, self.model[word])

        # get the average
        total_vector = numpy.divide(total_vector, len(review))
        return total_vector

    def process(self, texts_array, debug=False):
        """Given a set of reviews (each one a list of words), calculate
        the average feature vector for each one and return a numpy array
        """
        
        print("processing reviews using average")
        processed_texts = []
        # get the array size
        size = texts_array.shape[0]

        # Loop through the reviews
        for position in range(size):
            # visual feedback
            if(debug): progressBar.progressBar(position, size)
        
            # Call the function (defined above) that makes average feature vectors
            review = texts_array[position]
            processed_texts.append(self.average_review(review))

        processed_texts = numpy.array(processed_texts)
        print("\nprocessing done...\n")
        return processed_texts


"""
# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

trainDataVecs = getAvgFeatureVecs(clean_reviews, model, num_features)

#print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist(review, remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)"""
