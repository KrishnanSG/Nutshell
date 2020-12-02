# Nutshell
[![CodeFactor](https://www.codefactor.io/repository/github/krishnansg/nutshell/badge)](https://www.codefactor.io/repository/github/krishnansg/nutshell) 
[![Downloads](https://pepy.tech/badge/pynutshell)](https://pepy.tech/project/pynutshell)
[![Downloads](https://pepy.tech/badge/pynutshell/month)](https://pepy.tech/project/pynutshell)
[![PyPI version](https://badge.fury.io/py/pynutshell.svg)](https://pypi.org/project/pynutshell)

A simple to use yet robust python library containing tools to perform:

<img src="https://user-images.githubusercontent.com/43802499/99897377-02a9f300-2cbf-11eb-8830-d9bc8d2aa0d5.png"
align="right" 
     title="Nutshell" width="30%" height="30%">

- Text summarization
- Information retrieval
- Finding similarities 
- Sentence ranking
- Keyword extraction
- and many more in progress...



## Getting Started


These instructions will get you a copy of the project and ready for use for your python projects.

### Installation

  #### Quick Access
  - Download from PyPi.org
  
    ```bash
    pip install pynutshell
    ```
  
  #### Developer Style
  - Requires Python version >=3.6

  - Clone this repository using the command:

    ```bash
    git clone https://github.com/KrishnanSG/Nutshell.git
    cd Nutshell
    ```
    
  - Then install the library using the command:

    ```
      python setup.py install
    ```

> Note: The package is distributed as **pynutshell** due to unavailability of the name, but the package name is **nutshell** and request you not to get confused.


## How does the library work?

The library has several components:

1. Summarizers
2. Rankers
3. Similarity Algorithms
4. Information Retrievers
5. Keyword Extractors

### Summarization

A technique of transforming or condensing textual information using natural language processing techniques.


#### Types of summarization

<img src="https://user-images.githubusercontent.com/43802499/99897230-d5107a00-2cbd-11eb-8b7a-de7e3e0b7bc3.jpeg" 
align="right"
     title="Summarization" width="50%" height="30%">

**Extractive** 

This technique is very much similar to **highlighting important sentence** while we read a book.

The algorithm finds the important sentences in the corpus (NLP term for raw input text) by reducing the **similarity** between sentence by removing sentences which are very similar to each other by retaining one among them.

Though this method is a powerful it fails to combine 2 or more sentences into a complex sentence, there by not provide optimal result for some cases.


**Abstractive**

This technique unlike extractive is much more complex and robust in producing summaries. The algorithm used for this technique performs **sentence clustering** using **Semantic Analysis** (finding the meaning of sentence).


### Sentence Ranking

Text rankers are algorithms similar to web page ranking algorithms used to rank web pages. These rankers find the importance of the sentence in the document and provide ranks to the sentence, thereby providing us with the information of how important the sentence is.

### Similarity Algorithms

Text similarity algorithms define the similarity between 2 documents (sentences). 

A few classic algorithms for finding similarity are:
1. Cosine Similarity
2. Euclidean Distance

> Note: word2vec is an important transformation step used to convert words into vectors to easily perform mathematical operations.


## Features

Checklist of features the library currently offers and plans to offer.

- [x] Keyword Extraction
- [x] Text Tokenizers 
- [x] Text cleaners
- [ ] Semantic decoder
- Summarization
  - [x]  Extractive
  - [ ]  Abstractive
- Text Rankers
  - [x] Intermediate
  - [ ] Advanced
- Information Retrieval
  - [x] Intermediate
  - [ ] Advanced


## Examples

### Summarization

A simple example on how to use the library and perform extractive text summarization from the given input text(corpus).

```python
from nutshell.algorithms.information_retrieval import ClassicalIR
from nutshell.algorithms.ranking import TextRank
from nutshell.algorithms.similarity import BM25Plus
from nutshell.model import Summarizer
from nutshell.preprocessing.cleaner import NLTKCleaner
from nutshell.preprocessing.preprocessor import TextPreProcessor
from nutshell.preprocessing.tokenizer import NLTKTokenizer
from nutshell.utils import load_corpus, construct_sentences_from_ranking

# Example
corpus = load_corpus('input.txt')
print("\n --- Original Text ---\n")
print(corpus)

preprocessor = TextPreProcessor(NLTKTokenizer(), NLTKCleaner())
similarity_algorithm = BM25Plus()
ranker = TextRank()
ir = ClassicalIR()

# Text Summarization
model = Summarizer(preprocessor, similarity_algorithm, ranker, ir)
summarised_content = model.summarise(corpus, reduction_ratio=0.70, preserve_order=True)

print("\n --- Summarized Text ---\n")
print(construct_sentences_from_ranking(summarised_content))

```

### Keyword Extraction

A simple example on how to use the library and perform keyword extraction from the given input text(corpus).

```python
from nutshell.algorithms.information_retrieval import ClassicalIR
from nutshell.model import KeywordExtractor
from nutshell.preprocessing.cleaner import NLTKCleaner
from nutshell.preprocessing.preprocessor import TextPreProcessor
from nutshell.preprocessing.tokenizer import NLTKTokenizer
from nutshell.utils import load_corpus

corpus = load_corpus('input.txt')
print("\n --- Original Text ---\n")
print(corpus)

# Text Keyword Extraction
preprocessor = TextPreProcessor(NLTKTokenizer(), NLTKCleaner(skip_stemming=True))
keyword_extractor = KeywordExtractor(preprocessor, ClassicalIR())
keywords = keyword_extractor.extract_keywords(corpus, count=10, raw=False)


print("\n --- Keywords ---\n")
print(keywords)

```

## Contribution

Contributions are always welcomed, it would be great to have people use and contribute to this project to help user understand and benefit from library.

### How to contribute
- **Create an issue:** If you have a new feature in mind, feel free to open an issue and add some short description on what that feature could be.
- **Create a PR**: If you have a bug fix, enhancement or new feature addition, create a Pull Request and the maintainers of the repo, would review and merge them.

## Authors

* **Krishnan S G** - [@KrishnanSG](https://github.com/KrishnanSG)
* **Shruthi Abirami** - [@Shruthi-22](https://github.com/shruthi-22)
