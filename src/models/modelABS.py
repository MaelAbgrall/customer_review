# python integrated
import abc


class abstractModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def process(self):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError
