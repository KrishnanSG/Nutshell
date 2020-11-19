from abc import ABC, abstractmethod
from nltk.stem import WordNetLemmatizer 
from typing import List
from nutshell.preprocessing.tokenizer import Token
from nltk.corpus import stopwords

class BaseCleaner(ABC):
    """
    Interface for text cleaners
    """

    @staticmethod
    @abstractmethod
    def remove_stop_words(tokens: Token, *args) -> Token:
        pass

    @staticmethod
    @abstractmethod
    def to_lowercase(tokens: Token, *args) -> Token:
        pass

    @staticmethod
    @abstractmethod
    def remove_punctuation(tokens: Token, *args) -> Token:
        pass

    @staticmethod
    @abstractmethod
    def clean(tokens: Token, *arg) -> Token:
        pass

class NLTKCleaner(BaseCleaner):
    
    @staticmethod
    def remove_stop_words(tokens : Token) -> Token:
        stop_words = set(stopwords.words('english'))
        return Token(list(map(lambda token : [word for word in token if word not in stop_words] ,tokens.get_sentences())))

    @staticmethod
    def to_lowercase(tokens : Token) -> Token:
        return  Token(list(map(lambda token: [word.lower() for word in token], tokens.get_sentences())))

    @staticmethod
    def remove_punctuation(tokens : Token) -> Token:
        punctuation = tuple('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return  Token(list(map(lambda token: [word for word in token if word not in punctuation], tokens.get_sentences())))

    @staticmethod
    def lemmatize_words(tokens : Token) -> Token:
        lemmatizer = WordNetLemmatizer()
        return  Token(list(map(lambda token: [lemmatizer.lemmatize(word) for word in token], tokens.get_sentences() )))

    @staticmethod 
    def clean(tokens : Token) -> Token:
        
        lowercase_tokens = NLTKCleaner.to_lowercase(tokens)
        punctuation_removed_tokens = NLTKCleaner.remove_punctuation(lowercase_tokens)
        stop_words_removed_tokens = NLTKCleaner.remove_stop_words(punctuation_removed_tokens)
        lemmatized_sentences = NLTKCleaner.lemmatize_words(stop_words_removed_tokens)
        return stop_words_removed_tokens


