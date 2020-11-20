import os
from typing import Tuple, List


def load_corpus(file_path):
    corpus_file = os.path.abspath(file_path)
    with open(corpus_file) as f:
        corpus = f.read()
    return corpus


def construct_sentences_from_ranking(ranking: List[Tuple[float, List]]):
    text = []
    for score, sentence in ranking:
        text.append(' '.join(sentence))
    return '\n'.join(text)
