from nutshell.algorithms.information_retrieval import ClassicalIR
from nutshell.algorithms.ranking import BaseRanker
from nutshell.algorithms.similarity import BaseSimilarityAlgo
from nutshell.preprocessing.preprocessor import BasePreProcessor
from nutshell.preprocessing.tokenizer import Token


class Summarizer:
    def __init__(self, corpus, preprocessor: BasePreProcessor, similarity_algo: BaseSimilarityAlgo, ranker: BaseRanker):
        self.corpus = corpus
        self.preprocessor = preprocessor
        self.similarity_algo = similarity_algo
        self.ranker = ranker

    def summarise(self, ratio=0.5):
        """
        Returns the summarised the text based on given reduction ratio
        :param ratio: Reduction ratio
        :return: Summarised text
        """
        pass


def extract_keywords(tokens: Token, count=5, raw=False):
    tf = ClassicalIR.calculate_tf(tokens)
    idf = ClassicalIR.calculate_idf(tokens)
    keywords = ClassicalIR.cumulative_weight(tf, idf, order=True)
    return dict(zip(*keywords)) if raw else list(zip(*keywords))[0][:count]
