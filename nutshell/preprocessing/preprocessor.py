from typing import Tuple

from nutshell.preprocessing.cleaner import BaseCleaner, NLTKCleaner
from nutshell.preprocessing.tokenizer import BaseTokenizer, Token, NLTKTokenizer


class TextPreProcessor:

    def __init__(self, tokenizer: BaseTokenizer = NLTKTokenizer(), cleaner: BaseCleaner = NLTKCleaner()):
        """
        TextPreprocessor class is responsible performing tokenization and apply transforms using the cleaner.

        :param tokenizer: Tokenizer object which preforms text tokenization. By default NLTKTokenizer is used.
        :param cleaner:  Cleaner object which performs cleaning methods on the tokens. By default NLTKCleaner() is used.
        """
        self.__cleaner = cleaner
        self.__tokenizer = tokenizer

    def __repr__(self):
        return f"""TextPreProcessor(cleaner={self.__cleaner}, tokenizer={self.__tokenizer})"""

    def preprocess(self, corpus) -> Tuple[Token, Token]:
        """
        Preprocesses the corpus by invoking the tokenize and clean methods
        :return: Tokens and cleaned tokens
        """
        original_tokens = self.__tokenizer.tokenize(corpus)
        return original_tokens, self.__cleaner.clean(original_tokens)
