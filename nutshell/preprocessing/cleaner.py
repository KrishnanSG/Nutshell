from abc import ABC, abstractmethod

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nutshell.preprocessing.tokenizer import Token


class BaseCleaner(ABC):
    """
    Interface for text cleaners
    """

    @staticmethod
    @abstractmethod
    def remove_stop_words(*args) -> Token:
        pass

    @staticmethod
    @abstractmethod
    def to_lowercase(*args) -> Token:
        pass

    @staticmethod
    @abstractmethod
    def remove_punctuation(*args) -> Token:
        pass

    @staticmethod
    @abstractmethod
    def clean(*arg) -> Token:
        pass


class NLTKCleaner(BaseCleaner):

    def __init__(self, skip_stemming=False):
        self.__skip_stemming = skip_stemming

    @staticmethod
    def remove_stop_words(tokens: Token, language='english') -> Token:
        stop_words = set(stopwords.words(language))
        return Token(
            list(map(lambda token: [word for word in token if word not in stop_words], tokens.get_sentences())))

    @staticmethod
    def to_lowercase(tokens: Token) -> Token:
        return Token(list(map(lambda token: [word.lower() for word in token], tokens.get_sentences())))

    @staticmethod
    def remove_punctuation(tokens: Token) -> Token:
        punctuation = tuple('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return Token(
            list(map(lambda token: [word for word in token if word not in punctuation], tokens.get_sentences())))

    @staticmethod
    def stem_words(tokens: Token) -> Token:
        stemmer = PorterStemmer()
        return Token(list(map(lambda token: [stemmer.stem(word) for word in token], tokens.get_sentences())))

    def clean(self, tokens: Token) -> Token:
        result = NLTKCleaner.to_lowercase(tokens)
        result = NLTKCleaner.remove_punctuation(result)
        result = NLTKCleaner.remove_stop_words(result)
        if not self.__skip_stemming:
            result = NLTKCleaner.stem_words(result)
        return result
