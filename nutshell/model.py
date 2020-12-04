from nutshell.algorithms.information_retrieval import ClassicalIR
from nutshell.algorithms.ranking import BaseRanker, TextRank
from nutshell.algorithms.similarity import BaseSimilarityAlgo, BM25Plus
from nutshell.preprocessing.cleaner import NLTKCleaner
from nutshell.preprocessing.preprocessor import TextPreProcessor


class Summarizer:
    def __init__(
            self, preprocessor: TextPreProcessor = TextPreProcessor(),
            similarity_algo: BaseSimilarityAlgo = BM25Plus(),
            ranker: BaseRanker = TextRank(),
            ir: ClassicalIR = ClassicalIR()
    ):
        """
        Summarizer helps to summarise a corpus with the given reduction ratio.

        :param preprocessor: Text preprocessor algorithm. Default - TextPreProcessor.
        :param similarity_algo: Algorithm to be used for finding similarity between docs. Default - BM25Plus.
        :param ranker: Ranking algorithm to be used to rank the docs. Default - TextRank.
        :param ir: Information retrieval algorithm to be used to extract tf, idf and other necessary measures.
            Default - ClassicalIR.
        """
        self.__preprocessor = preprocessor
        self.__similarity_algo = similarity_algo
        self.__ranker = ranker
        self.__ir = ir

    def __repr__(self):
        return f"""Summarizer(preprocessor={self.__preprocessor},
           similarity_algo={self.__similarity_algo},
           ranker={self.__ranker},
           ir={self.__ir}
        )"""

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
        original_token, cleaned_tokens = self.__preprocessor.preprocess(corpus)

        # Information retrieval
        _idf = self.__ir.calculate_idf(cleaned_tokens)

        # Similarity and Ranking
        similarity_matrix = self.__similarity_algo.similarity_matrix(cleaned_tokens, _idf)
        scores = self.__ranker.get_ranking_scores(similarity_matrix)

        summarized_content = self.__ranker.get_top(scores, original_token, reduction_ratio=reduction_ratio,
                                                   preserve_order=preserve_order)
        return summarized_content


class KeywordExtractor:
    def __init__(
            self,
            preprocessor: TextPreProcessor = TextPreProcessor(cleaner=NLTKCleaner(skip_stemming=True)),
            ir: ClassicalIR = ClassicalIR()
    ):
        """

        :param preprocessor: Text preprocessor algorithm. Default - TextPreProcessor.
        :param ir: Information retrieval algorithm to be used to extract tf, idf and other necessary measures.
            Default - ClassicalIR.
        """
        self.__preprocessor = preprocessor
        self.__ir = ir

    def __repr__(self):
        return f"KeywordExtractor(preprocessor={self.__preprocessor}, ir={self.__ir})"

    def extract_keywords(self, corpus, count=5, raw=False):
        original_token, tokens = self.__preprocessor.preprocess(corpus)
        tf = self.__ir.calculate_tf(tokens)
        idf = self.__ir.calculate_idf(tokens)
        keywords = ClassicalIR.cumulative_weight(tf, idf, order=not raw)
        return dict(zip(keywords.keys(), keywords.values())) if raw else list(zip(*keywords))[0][:count]
