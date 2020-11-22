from nutshell.algorithms.information_retrieval import ClassicalIR
from nutshell.algorithms.ranking import TextRank
from nutshell.algorithms.similarity import BM25Plus
from nutshell.model import Summarizer, KeywordExtractor
from nutshell.preprocessing.cleaner import NLTKCleaner
from nutshell.preprocessing.preprocessor import TextPreProcessor
from nutshell.preprocessing.tokenizer import NLTKTokenizer
from nutshell.utils import load_corpus, construct_sentences_from_ranking

corpus = load_corpus('input.txt')
print("\n --- Original Text ---\n")
print(corpus)

tokenizer = NLTKTokenizer()
similarity_algorithm = BM25Plus()
ranker = TextRank()
ir = ClassicalIR()

# Text Summarization
preprocessor = TextPreProcessor(tokenizer, NLTKCleaner())
model = Summarizer(preprocessor, similarity_algorithm, ranker, ir)
summarised_content = model.summarise(corpus, reduction_ratio=0.70, preserve_order=True)

print("\n --- Summarized Text ---\n")
print(construct_sentences_from_ranking(summarised_content))

# Text Keyword Extraction
preprocessor = TextPreProcessor(tokenizer, NLTKCleaner(skip_stemming=True))
keyword_extractor = KeywordExtractor(preprocessor, ClassicalIR())
keywords = keyword_extractor.extract_keywords(corpus, count=10, raw=False)

print("\n --- Keywords ---\n")
print(keywords)
