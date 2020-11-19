from nutshell.preprocessing.tokenizer import NLTKTokenizer
from nutshell.preprocessing.cleaner import NLTKCleaner
from nutshell.preprocessing.preprocessor import TextPreProcessor

corpus = """One morning, when Gregor Samsa woke from troubled dreams, he found
himself transformed in his bed into a horrible vermin.  He lay on
his armour-like back, and if he lifted his head a little he could
see his brown belly, slightly domed and divided by arches into stiff
sections.  The bedding was hardly able to cover it and seemed ready
to slide off any moment.  His many legs, pitifully thin compared
with the size of the rest of him, waved about helplessly as he
looked.
"""

if __name__ == "__main__":
    obj = TextPreProcessor(corpus, NLTKTokenizer, NLTKCleaner)
    op = obj.preprocess()

    print('Original sentences')
    for i in op['original'].get_sentences():
        print( i )
    print('\n')

    print('After cleaning (lexicalisation): ')
    for i in op['cleaned'].get_sentences():
        print(  i )
    print('\n')