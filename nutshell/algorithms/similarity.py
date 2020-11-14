from abc import ABC, abstractmethod
from math import log
from pprint import pprint
from typing import Dict

import numpy as np

from nutshell.preprocessing.tokenizer import Token


class BaseSimilarityAlgo(ABC):
    @abstractmethod
    def _calculate_similarity_score(self, doc1: list, doc2: list) -> float:
        pass

    @abstractmethod
    def similarity_matrix(self, *args) -> Dict[int, int]:
        pass


class BM25Plus(BaseSimilarityAlgo):
    """
    BM25Plus is an algorithm to find similarity b/w 2 docs/sentences
    """

    def __init__(self, tokens: Token, k1: float = 1.2, b: float = 0.75):
        self.__tokens = tokens
        self.__avg_doc_len = self.__tokens.get_avg_token_per_sentence()

        # Algorithm specific parameters
        self.__k1 = k1
        self.__b = b

    @staticmethod
    def calculate_doc_word_freq(word, documents, return_count=True) -> int:
        """
        Returns the number of the documents that contain the given word
        """
        docs_containing_word = list(filter(lambda x: word in x, documents))
        return len(docs_containing_word) if return_count else docs_containing_word

    def calculate_idf(self) -> Dict[str, float]:
        """
        idf = 1 + log(total number of docs/number of docs containing the word)
        :return: IDF
        """
        idf = {}
        word_doc_freq = {}
        for sen in self.__tokens.get_sentences():
            for word in sen:
                if word_doc_freq.get(word) is None:
                    word_doc_freq[word] = BM25Plus.calculate_doc_word_freq(word, self.__tokens.get_sentences())
                idf[word] = 1 + log(self.__tokens.get_number_of_sentences() / word_doc_freq[word])
        return idf

    def _calculate_similarity_score(self, doc1: list, doc2: list) -> float:
        """
        Calculates the similarity score b/w given 2 docs
        :param doc1: Document 1
        :param doc2: Document 2
        :return: similarity score
        """
        score = 0
        for word in doc2:
            freq_of_word_in_doc1 = doc1.count(word)
            score += freq_of_word_in_doc1 * (self.__k1 + 1) / (freq_of_word_in_doc1 + self.__k1 * (
                    1 - self.__b + self.__b * len(doc1) / self.__avg_doc_len))
        return score

    def similarity_matrix(self) -> np.ndarray:
        """
        Calculates the similarity matrix for the docs
        :return: similarity matrix
        """
        n = self.__tokens.get_number_of_sentences()
        matrix = np.zeros((n, n))
        for i, doc1 in enumerate(self.__tokens.get_sentences()):
            for j, doc2 in enumerate(self.__tokens.get_sentences()):
                if i != j:
                    matrix[i][j] = self._calculate_similarity_score(doc1, doc2)
        return matrix