from collections import defaultdict
from math import log
from typing import Dict

from nutshell.preprocessing.tokenizer import Token


class ClassicalIR:

    def __repr__(self):
        return f"ClassicalIR()"

    @staticmethod
    def calculate_tf(tokens: Token) -> Dict[str, Dict[str, float]]:
        """
        tf = (frequency of word in doc/number of tokens in that doc)
        :param tokens: Tokens for the corpus
        :return: TF
        """
        tf = defaultdict(dict)
        for idx, doc in enumerate(tokens.get_sentences()):
            for word in doc:
                tf[f'doc{idx}'][word] = doc.count(word) / len(doc)
        return tf

    @staticmethod
    def __calculate_doc_word_freq(word, documents, return_count=True) -> int:
        """
        Returns the number of the documents that contain the given word
        """
        docs_containing_word = list(filter(lambda x: word in x, documents))
        return len(docs_containing_word) if return_count else docs_containing_word

    @staticmethod
    def calculate_idf(tokens: Token) -> Dict[str, float]:
        """
        idf = 1 + log(total number of docs/number of docs containing the word)
        :param tokens: Tokens for the corpus
        :return: IDF
        """
        idf = {}
        word_doc_freq = {}
        for sen in tokens.get_sentences():
            for word in sen:
                if word_doc_freq.get(word) is None:
                    word_doc_freq[word] = ClassicalIR.__calculate_doc_word_freq(word, tokens.get_sentences())
                idf[word] = 1 + log(tokens.get_number_of_sentences() / word_doc_freq[word])
        return idf

    @staticmethod
    def calculate_weight(tf: Dict[str, Dict[str, float]], idf: Dict[str, float]):
        """
        Calculates weight of word across in each docs
        :param tf: Term frequency
        :param idf: Inverse document frequency
        :return: Returns weight of all words per doc
        """
        weight = defaultdict(dict)
        for doc_id, doc in tf.items():
            for word in doc:
                weight[doc_id][word] = tf[doc_id][word] * idf[word]
        return weight

    @staticmethod
    def cumulative_weight(tf: Dict[str, Dict[str, float]], idf: Dict[str, float], order=False):
        """
        Calculates cumulative weight of word across all docs
        :param tf: Term frequency
        :param idf: Inverse document frequency
        :param order: If True, the words returned are sort in desc based on the weights
        :return: Returns cumulative weight of all words
        """
        weight = defaultdict()
        for doc_id, doc in tf.items():
            for word in doc:
                if weight.get(word) is None:
                    weight[word] = 0
                weight[word] += tf[doc_id][word] * idf[word]

        return sorted(weight.items(), key=lambda x: x[1], reverse=True) if order else weight
