# Sequence Tagging
## Comparing Language Models
This Project implement Trigram HMM to tag word sequence.

## Getting Started

### Usage
```
$ python data.py
```

*count_freq.py* : The primary file to run. This file contains methods to read the appropriate data files from the archive, training baseline tagger and viterbi model, then creat a file that is able to be evaluated by the evaluation function.
>>> python count_freq.py gene.train gene_dev.p1.out

*eval_gene_tagger.py*: This file contains functions to evaluate the predictive tag, including F1-score, recall, and precision
>>eval_gene_tagger.py gene.key gene_dev.p1.out

