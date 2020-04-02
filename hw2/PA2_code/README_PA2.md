# CSE256_PA2

# CSE256_NLP PA1 project
## Programming Assignment 2: Semi-supervised Text Classification
This project first carry out supervised learning with training data, trying to tuning hyperparameters on models. Then use the model obtained to do semi-supervised learning on unlabeled data.Finally, compared and analyze the results.

## Getting Started

### Usage
```
$ python PA2.py
```

*PA2.py* : The primary file to run. This file contains methods to train classifier and evaluate the model. And get stopwords from dataset. The main function do supervised learning and semi-supervised learning, then further analysize the result.

*sentiment_new.py*: This file includes method to read training and unlabeled data, writing kaggle file, and integrate training and unlabeled dataset
