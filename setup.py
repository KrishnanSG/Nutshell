from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pynutshell',
    version='1.0.2',
    packages=find_packages(exclude=['tests', 'tests.*', '.github']),
    install_requires=[
        'numpy',
        'networkx',
        'nltk'
    ],
    url='https://github.com/KrishnanSG/Nutshell',
    author='Krishnan S G, Shruthi Abirami',
    description="An unsupervised text summarization and information retrieval library under the hood using natural "
                "language processing models.",
    author_email='krishsg525@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="text, summarization, nutshell, ranking, information, retrieval, similarity, keyword, extraction, nlp, "
             "natural, language, nltk, tokenizer, pagerank, textrank, bm25plus",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
