# CSE256_NLP PA1 project
## Comparing Language Models
This Project implement trigram bigram, unigram models with add-one/Laplace smoothing, and make comparison of perplexity between language models by tuning hyperparameters.

## Getting Started

### Prerequisites
The one optional moedule in this code is `tabulate` ([documentation](https://pypi.python.org/pypi/tabulate)).
This package is quite useful for generating the results table in LaTeX directly from python code. If you do not install this package, the code does not write out the results to file (there's no runtime error).

### Usage
```
$ python data.py
```

*data.py* : The primary file to run. This file contains methods to read the appropriate data files from the archive, train and evaluate all the trigram language models (by calling “lm.py”), and generate sample sentences from all the models (by calling  “generator.py”). It also saves the result tables into LaTeX files.

*lm.py*: This file describes the higher level interface for a language model, and contains functions to train, query, and evaluate it. An implementation of a simple back-off based unigram model is also included, that implements all of the functions
of the interface.

*generator.py*: This file contains a simple word and sentence sampler for any language model. Since it supports arbitarily complex language models, it is not very efficient. If this sampler is incredibly slow for your language model, you can consider implementing your own (by caching the conditional probability tables, for example, instead of computing it for every word).
