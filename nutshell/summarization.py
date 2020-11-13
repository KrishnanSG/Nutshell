from nutshell.preprocessing.preprocessor import BasePreProcessor


class Nutshell:
    def __init__(self, corpus, preprocessor: BasePreProcessor):
        self.corpus = corpus
        self.preprocessor = preprocessor

    def summarise(self, ratio=0.5):
        """
        Returns the summarised the text based on given reduction ratio
        :param ratio: Reduction ratio
        :return: Summarised text
        """
        pass
