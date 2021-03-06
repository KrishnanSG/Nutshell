from abc import ABC, abstractmethod
from typing import List, Union, Iterable

import nltk


class Token:
    def __init__(self, raw_tokens: List[List[str]]):
        self.__raw_tokens = raw_tokens

    def get_sentence(self, sentence_id) -> List:
        if 0 <= sentence_id <= len(self.__raw_tokens) - 1:
            return self.__raw_tokens[sentence_id]
        raise Exception(f"Invalid sentence Id. Valid sentence ids are of range [0, {len(self.__raw_tokens) - 1}]")

    def get_sentences(self):
        for sentence in self.__raw_tokens:
            yield sentence

    def get_number_of_tokens(self, raw=True) -> Union[List, int]:
        """
        Returns the number of __tokens per sentence
        :param raw: If false returns the total number of __tokens
        :return: List of __tokens per sentence (1*n) -> n number of sentences
        """
        count = list(map(len, self.get_sentences()))
        return count if raw else sum(count)

    def get_avg_token_per_sentence(self) -> float:
        res = self.get_number_of_tokens()
        return sum(res) / len(res)

    def get_number_of_sentences(self):
        return len(self.__raw_tokens)


class BaseTokenizer(ABC):
    """
    Interface for text tokenizers
    """
    @abstractmethod
    def __repr__(self):
        pass

    @staticmethod
    @abstractmethod
    def tokenize_into_words(*args) -> Iterable[str]:
        pass

    @staticmethod
    @abstractmethod
    def tokenize_into_sentences(*args) -> Iterable[str]:
        pass

    @staticmethod
    @abstractmethod
    def tokenize(*args) -> Token:
        pass


class NLTKTokenizer(BaseTokenizer):

    def __repr__(self):
        return f"""NLTKTokenizer()"""

    @staticmethod
    def tokenize_into_words(sentence: str) -> List[str]:
        return nltk.word_tokenize(sentence)

    @staticmethod
    def tokenize_into_sentences(corpus: str) -> List[str]:
        return nltk.sent_tokenize(corpus)

    @staticmethod
    def tokenize(corpus: str) -> Token:
        """Tokenizes the given corpus into sentences and words"""
        sentences = NLTKTokenizer.tokenize_into_sentences(corpus)
        raw_tokens = list(map(NLTKTokenizer.tokenize_into_words, sentences))
        return Token(raw_tokens)
