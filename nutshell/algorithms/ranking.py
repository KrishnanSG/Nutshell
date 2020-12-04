from abc import abstractmethod, ABC
from heapq import nlargest
from math import ceil
from typing import Dict, Any, List

import networkx as nx
import numpy as np

from nutshell.preprocessing.tokenizer import Token


class BaseRanker(ABC):

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def _ranking_algorithm(self, *args, **kwargs) -> Dict[Any, float]:
        pass

    def get_ranking_scores(self, similarity_matrix: np.ndarray) -> Dict[Any, float]:
        """
        Invokes the ranking algorithms and returns the rankings scores for each doc/sentence
        """
        return self._ranking_algorithm(similarity_matrix)

    @staticmethod
    @abstractmethod
    def get_top(*args, **kwargs) -> List:
        pass


class TextRank(BaseRanker):

    def __repr__(self):
        return f"TextRank()"

    def _ranking_algorithm(self, similarity_matrix: np.ndarray):
        """
        Calculates doc ranking using pagerank algorithm
        :return: Ranking scores for each doc/sentence
        """
        graph = nx.MultiDiGraph(similarity_matrix)
        return nx.pagerank_numpy(graph)

    @staticmethod
    def get_top(scores: dict, tokens: Token, reduction_ratio=0.70, preserve_order=False):
        """
        Returns the top n doc/sentences based on the reduction_ration
        :param scores: Ranking scores, computed using the ranking algorithm
        :param tokens: Docs/Sentences used for the model
        :param reduction_ratio: Reduction ratio expected for the output text. i.e if ratio=0.5 then half the number
                of sentence are returned
        :param preserve_order: If True, then sentence order is preserved
        :return: Top n sentences
        """
        p = tokens.get_number_of_sentences()
        n = ceil(p * (1 - reduction_ratio))
        print(f"\n --- Stats ---\nNumber of sentences before summarization: {p}\n"
              f"Number of sentences after summarization: {int(n)}")

        if preserve_order:
            # Return the sentences as it was in the original corpus without disturbing the order
            threshold = nlargest(n, scores.items(), key=lambda x: x[1])[-1][1]
            return [(scores[idx], sen) for idx, sen in enumerate(tokens.get_sentences()) if scores[idx] >= threshold]

        sentences = list(sorted(((scores[i], s) for i, s in enumerate(tokens.get_sentences())), reverse=True))
        return sentences[:n]
