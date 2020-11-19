from abc import ABC, abstractmethod

from nutshell.preprocessing.cleaner import BaseCleaner
from nutshell.preprocessing.tokenizer import BaseTokenizer


class TextPreProcessor():

    def __init__(self, corpus, tokenizer: BaseTokenizer, cleaner: BaseCleaner):
        self._corpus = corpus
        self._cleaner = cleaner
        self._tokenizer = tokenizer

    def preprocess(self, *args):
        
        original = self._tokenizer.tokenize(self._corpus)
        return {'original' : original, 'cleaned' : self._cleaner.clean(original) }

        
        
