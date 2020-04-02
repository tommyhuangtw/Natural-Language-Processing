
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:27:45 2019

@author: tommyhuang
"""

import sentiment_new as sentimentinterface
#import classify 
import timeit
import numpy as np

import matplotlib.pyplot as plt
#lt.switch_backend('agg')
import matplotlib.ticker as ticker

def train_classifier(X, y, penalty = 'l1', solver='lbfgs'):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression(random_state=0, penalty=penalty, solver=solver, max_iter=100000)
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls, name = 'data'):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    return acc
    #print("  Accuracy on %s  is: %s" % (name, acc))


def get_stopwords(n,fea_names,coeffs):
    '''
    get stopwords from feature names given cls.coefficients

    Args:
        fea_names = sentiment.count_vect.get_feature_names()
        coeffs = cls.coef_[0]
    Return:
        List of non-useful words

    '''
    assert (n%2==0) and isinstance(n,int)
    k=int(n/2)
    sortword =np.argsort(coeffs)
    top_k= sortword[-k:]
    bottom_k=sortword[:k]
    useful_words=[]
    for i in top_k:
        useful_words.append(fea_names[i])

    for i in bottom_k:
        useful_words.append(fea_names[i])

    stopwords= list(set(fea_names)-set(useful_words))
    return stopwords


def get_useful_words(n,fea_names,coeffs):
    '''
    get importtant words from feature names given cls.coefficients

    Args:
        fea_names = sentiment.count_vect.get_feature_names()
        coeffs = cls.coef_[0]
    Return:
        List of important words

    '''
    assert (n%2==0) and isinstance(n,int)
    k=int(n/2)
    sortword =np.argsort(coeffs)
    top_k= sortword[-k:]
    bottom_k=sortword[:k]
    toxic_words=[]
    nontoxic_words=[]
    for i in top_k:
        toxic_words.append(fea_names[i])

    for i in bottom_k:
        nontoxic_words.append(fea_names[i])

    return toxic_words , nontoxic_words


if __name__ == "__main__":
    print("Reading data")
    #sentiment = sentimentinterface.read_files(tarfname)
    
    #for mindf in np.linspace(1,20,20,dtype = int):
    #for maxdf in np.linspace(0.25,0.35,20):
    maxdf = 0.32
    #maxdf = 0.0371
    mindf = 1

    datanum = 4000

    sentiment = sentimentinterface.read_files( min_df = mindf, max_df = maxdf)
    #array = sentiment.trainX # the vocabulary dictionary and return term-document matrix

    solve_name = 'sag'
    penalty = 'l2'
    print("finished reading data")
    print("start training data")

    cls = train_classifier(sentiment.trainX, sentiment.trainy, penalty = penalty, solver = solve_name)
    print("Classifier trained")

    coef_now = cls.coef_[0]
    print('shape of coef_pre:' ,coef_now.shape)

    toxic_words, nontoxic_words = get_useful_words(50,fea_names,coeffs)
    print("________USEFUL WORDS_________")
    print("toxic word list: ", toxic_words)
    print("nontoxic word list: ", nontoxic_words)

    #stoplist = get_stopwords(datanum,sentiment.count_vect.get_feature_names(),cls.coef_[0])
    acc = evaluate(sentiment.devX, sentiment.devy, cls, 'dev data')

    #print('before using stopwords' )
    print('Accuracy before using stopwords: {}'.format(acc))

