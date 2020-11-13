from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """
    Interface for text tokenizers
    """

    @staticmethod
    @abstractmethod
    def tokenize(corpus, *args):
        pass

    @staticmethod
    @abstractmethod
    def tokenize_into_sentences(corpus, *args):
        pass
