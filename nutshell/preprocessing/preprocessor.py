from typing import Tuple

from nutshell.preprocessing.cleaner import BaseCleaner
from nutshell.preprocessing.tokenizer import BaseTokenizer, Token


class TextPreProcessor:

    def __init__(self, tokenizer: BaseTokenizer, cleaner: BaseCleaner):
        self.__cleaner = cleaner
        self.__tokenizer = tokenizer

    def preprocess(self, corpus) -> Tuple[Token, Token]:
        """
        Preprocesses the corpus by invoking the tokenize and clean methods
        :return: Tokens and cleaned tokens
        """
        original_tokens = self.__tokenizer.tokenize(corpus)
        return original_tokens, self.__cleaner.clean(original_tokens)
