from abc import ABC, abstractmethod

from nutshell.preprocessing.cleaner import BaseCleaner
from nutshell.preprocessing.tokenizer import BaseTokenizer


class BasePreProcessor(ABC):

    def __init__(self, corpus, tokenizer: BaseTokenizer, cleaner: BaseCleaner):
        self._corpus = corpus
        self._cleaner = cleaner
        self._tokenizer = tokenizer

    @abstractmethod
    def preprocess(self, *args):
        pass
