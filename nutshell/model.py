from nutshell.algorithms.information_retrieval import ClassicalIR
from nutshell.algorithms.ranking import BaseRanker
from nutshell.algorithms.similarity import BaseSimilarityAlgo
from nutshell.preprocessing.preprocessor import TextPreProcessor


class Summarizer:
    def __init__(self, preprocessor: TextPreProcessor, similarity_algo: BaseSimilarityAlgo, ranker: BaseRanker,
                 ir: ClassicalIR):
        self.preprocessor = preprocessor
        self.similarity_algo = similarity_algo
        self.ranker = ranker
        self.ir = ir

    def summarise(self, corpus, reduction_ratio=0.70, preserve_order=False):
        """
        Returns the summarised the text based on given reduction ratio
        :param corpus: Text to be summarized
        :param reduction_ratio: Reduction ratio expected for the output text. i.e if ratio=0.5 then half the number
                of sentence are returned
        :param preserve_order: If True, then sentence order is preserved
        :return: Summarised text
        """
        # Model Pipeline

        # Preprocessing
        original_token, cleaned_tokens = self.preprocessor.preprocess(corpus)

        # Information retrieval
        _idf = self.ir.calculate_idf(cleaned_tokens)

        # Similarity and Ranking
        similarity_matrix = self.similarity_algo.similarity_matrix(cleaned_tokens, _idf)
        scores = self.ranker.get_ranking_scores(similarity_matrix)

        summarized_content = self.ranker.get_top(scores, original_token, reduction_ratio=reduction_ratio,
                                                 preserve_order=preserve_order)
        return summarized_content


class KeywordExtractor:
    def __init__(self, preprocessor: TextPreProcessor, ir: ClassicalIR):
        self.preprocessor = preprocessor
        self.ir = ir

    def extract_keywords(self, corpus, count=5, raw=False, ):
        original_token, tokens = self.preprocessor.preprocess(corpus)
        tf = self.ir.calculate_tf(tokens)
        idf = self.ir.calculate_idf(tokens)
        keywords = ClassicalIR.cumulative_weight(tf, idf, order=not raw)
        return dict(zip(keywords.keys(), keywords.values())) if raw else list(zip(*keywords))[0][:count]
