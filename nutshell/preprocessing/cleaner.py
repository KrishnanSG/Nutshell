from abc import ABC, abstractmethod

from nutshell.preprocessing.tokenizer import Token


class BaseCleaner(ABC):
    """
    Interface for text cleaners
    """

    @staticmethod
    @abstractmethod
    def remove_stop_words(tokens: Token, *args):
        pass

    @staticmethod
    @abstractmethod
    def clean(tokens: Token, *arg):
        pass
