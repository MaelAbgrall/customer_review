# python integrated

# dependencies

# project files
import models.modelABS as modelABS
import utils.fileHandler as fileHandler


class MLModel(modelABS.abstractModel):
    """the Idea of this class is to provide a clean interface to sklearn models as well as keras models. 
    It is not the best way to tweak an ML model 
    (best thing to do is to create and tweak a model elsewhere, save it and then use this class as a helper)
    """

    def __init__(self, model_type, path=None):
        if(path is not None):
            self.model = fileHandler.load_pickle(path)
        
        self.model_type = model_type

        # TODO:
        # TODO:
        # TODO:
        # TODO:
        # TODO:
        # TODO:
        # TODO:
        # TODO:# TODO:
        # s'occuper de faire marcher word2vec 
        # avant de s'occuper des models!


