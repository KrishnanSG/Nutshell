from abc import ABC, abstractmethod

from nutshell.preprocessing.cleaner import BaseCleaner
from nutshell.preprocessing.tokenizer import BaseTokenizer


class BasePreProcessor(ABC):

    def __init__(self, corpus, tokenizer: BaseTokenizer, cleaner: BaseCleaner):
        self.__corpus = corpus
        self.__cleaner = cleaner
        self.__tokenizer = tokenizer

    @abstractmethod
    def preprocess(self, *args):
        pass
