from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from nutshell.preprocessing.tokenizer import Token


class BaseSimilarityAlgo(ABC):
    @abstractmethod
    def _calculate_similarity_score(self, doc1: list, doc2: list) -> float:
        pass

    @abstractmethod
    def similarity_matrix(self, *args) -> np.ndarray:
        pass


class BM25Plus(BaseSimilarityAlgo):
    """
    BM25Plus is an algorithm to find similarity b/w 2 docs/sentences
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.__idf = None
        self.__avg_doc_len: float = 0

        # Algorithm specific parameters
        self.__k1 = k1
        self.__b = b

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
            score += self.__idf[word] * freq_of_word_in_doc1 * (self.__k1 + 1) / (freq_of_word_in_doc1 + self.__k1 * (
                    1 - self.__b + self.__b * len(doc1) / self.__avg_doc_len))
        return score

    def similarity_matrix(self, tokens: Token, idf: Dict[str, float]) -> np.ndarray:
        """
        Calculates the similarity matrix for the docs
        :return: similarity matrix
        """
        self.__avg_doc_len = tokens.get_avg_token_per_sentence()
        self.__idf = idf
        n = tokens.get_number_of_sentences()
        matrix = np.zeros((n, n))
        for i, doc1 in enumerate(tokens.get_sentences()):
            for j, doc2 in enumerate(tokens.get_sentences()):
                if i != j:
                    matrix[i][j] = self._calculate_similarity_score(doc1, doc2)
        return matrix
