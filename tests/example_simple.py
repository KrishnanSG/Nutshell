"""
Here is an example of the output produced by the summarization and keyword extraction module

Original Text
------------

With only 1000 fine-tuning examples, we were able to perform better in most tasks than a strong baseline
(Transformer encoder-decoder) that used the full supervised data, which in some cases had many orders of
magnitude more examples.
This “sample efficiency” greatly increases the usefulness of text summarization models as it significantly lowers
the scale and cost of supervised data collection, which in the case of summarization is very expensive.
While we find automatic metrics such as ROUGE are useful proxies for measuring progress during model development,
they only provide limited information and don’t tell us the whole story, such as fluency or a comparison
to human performance.
To this end, we conducted a human evaluation, where raters were asked to compare summaries from our model with
human ones (without knowing which is which). This has some similarities to the Turing test.
Following this post is an example article from the XSum dataset along with the model-generated abstractive summary.
The model correctly abstracts and paraphrases four named frigates (HMS Cumberland, HMS Campbeltown, HMS Chatham and
HMS Cornwall) as “four Royal Navy frigates”, something an extractive approach could not do since “four” is not
mentioned anywhere. Was this a fluke or did the model actually count? One way to find out is to add and remove ships
to see if the count changes.As can be seen below, the model successfully “counts” ships from 2 to 5.
However, when we add a sixth ship, the “HMS Alphabet”, it miscounts it as “seven”.
So it appears the model has learned to count small numbers of items in a list, but does not yet generalize as elegantly
as we would hope. Still, we think this rudimentary counting ability is impressive as it was not explicitly programmed
into the model, and it demonstrates a limited amount of “symbolic reasoning” by the model.

Summarized Text
--------------

This “sample efficiency” greatly increases the usefulness of text summarization models as it significantly lowers
the scale and cost of supervised data collection , which in the case of summarization is very expensive.While we find
automatic metrics such as ROUGE are useful proxies for measuring progress during model development , they only provide
limited information and don ’ t tell us the whole story , such as fluency or a comparison to human performance .
The model correctly abstracts and paraphrases four named frigates ( HMS Cumberland , HMS Campbeltown , HMS Chatham and
HMS Cornwall ) as “ four Royal Navy frigates ” , something an extractive approach could not do since “ four ”
is not mentioned anywhere.
One way to find out is to add and remove ships to see if the count changes. As can be seen below,
the model successfully “counts” ships from 2 to 5.

Keywords
--------

'similarities', 'turing', 'test', 'model', 'count', 'fluke', 'actually', 'pegasus', 'hms',
'human', 'code', 'add', 'ships', 'thank', 't5'

"""
from nutshell.model import Summarizer, KeywordExtractor
from nutshell.utils import load_corpus, construct_sentences_from_ranking

corpus = load_corpus('sample.txt')
print("\n --- Original Text ---\n")

print(corpus)

model = Summarizer()
summarised_content = model.summarise(corpus, reduction_ratio=0.80, preserve_order=False)

print("\n --- Summarized Text ---\n")
print(construct_sentences_from_ranking(summarised_content))

keyword_extractor = KeywordExtractor()
keywords = keyword_extractor.extract_keywords(corpus, count=17, raw=False)

print("\n --- Keywords ---\n")
print(keywords)
