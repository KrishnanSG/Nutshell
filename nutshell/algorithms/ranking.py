from abc import abstractmethod, ABC
from math import ceil
from pprint import pprint
from typing import Dict, Any, List

import networkx as nx
import numpy as np

from nutshell.algorithms.similarity import BM25Plus
from nutshell.preprocessing.tokenizer import Token


class BaseRanker(ABC):

    @abstractmethod
    def _ranking_algorithm(self, *args, **kwargs) -> Dict[Any, float]:
        pass

    def get_ranking_scores(self) -> Dict[Any, float]:
        """
        Invokes the ranking algorithms and returns the rankings scores for each doc/sentence
        """
        return self._ranking_algorithm()

    @staticmethod
    @abstractmethod
    def get_top(*args, **kwargs) -> List:
        pass


class TextRank(BaseRanker):

    def __init__(self, similarity_matrix: np.ndarray):
        self.__similarity_matrix = similarity_matrix

    def _ranking_algorithm(self):
        """
        Calculates doc ranking using pagerank algorithm
        :return: Ranking scores for each doc/sentence
        """
        graph = nx.MultiDiGraph(self.__similarity_matrix)
        return nx.pagerank_numpy(graph)

    @staticmethod
    def get_top(scores, tokens: Token, reduction_ratio=0.70, preserve_order=False):
        """
        Returns the top n doc/sentences based on the reduction_ration
        :param scores: Ranking scores, computed using the ranking algorithm
        :param tokens: Docs/Sentences used for the model
        :param reduction_ratio: Reduction ratio expected for the output text. i.e if ratio=0.5 then half the number
                of sentence are returned
        :param preserve_order: If True, then sentence order is preserved
        :return:
        """
        if preserve_order:
            # TODO: Add logic to preserve order and return the sentences
            pass
        else:
            sentences = list(sorted(((scores[i], s) for i, s in enumerate(tokens.get_sentences())), reverse=True))
            n = ceil(tokens.get_number_of_sentences() * reduction_ratio)
            return sentences[:n]


if __name__ == '__main__':
    corpus = [
        ["Hello", "there", "good", "man!"],
        ["It", "is", "quite", "windy", "in", "London"],
        ["Hello", "How", "is", "the", "weather", "today?"],
    ]
    tokens = Token(corpus)
    bm25plus = BM25Plus(tokens)

    print("Similarity Matrix")
    mat = bm25plus.similarity_matrix()
    pprint(mat)

    tr = TextRank(mat)
    scores = tr.get_ranking_scores()

    print("Ranking Scores")
    print(scores)
    pprint(tr.get_top(scores, tokens, reduction_ratio=0.5))
