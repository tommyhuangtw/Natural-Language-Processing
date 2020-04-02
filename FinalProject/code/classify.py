#!/bin/python
import pickle
def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	c_value = 10
	# print("C: ", c_value)
	cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000, C=c_value)
	cls.fit(X, y)
	with open('model.pkl', 'wb') as fout:
		pickle.dump(cls, fout)
	print("finish dumping model_coef")
	return cls

def evaluate(X, yt, cls, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print("  Accuracy on %s  is: %s" % (name, acc))
