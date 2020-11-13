from abc import ABC, abstractmethod


class BaseCleaner(ABC):
    """
    Interface for text cleaners
    """

    @staticmethod
    @abstractmethod
    def remove_stop_words(tokens, *args):
        pass

    @staticmethod
    @abstractmethod
    def clean(tokens, *arg):
        pass
