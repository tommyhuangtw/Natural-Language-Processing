
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



if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    #sentiment = sentimentinterface.read_files(tarfname)
    
    #for mindf in np.linspace(1,20,20,dtype = int):
    #for maxdf in np.linspace(0.25,0.35,20):
    maxdf = 0.32
    #maxdf = 0.0371
    mindf = 1
    #mindf=32
    #maxfeas = 10000
    #for maxfeas in np.linspace(1000, 10000, 10, dtype=int):
    #sentiment = sentimentinterface.read_files(tarfname, min_df = 1, max_df =1.0, max_features= maxfeas)
    #acc_dict=dict()
    #for datanum in np.linspace(4000,9000,3,dtype = int):
    #for maxdf in np.linspace(0.004,3,10):
    datanum = 4000
    ngram_list=[(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(3,4),(4,4)]

    sentiment = sentimentinterface.read_files(tarfname, min_df = mindf, max_df = maxdf)
    #array = sentiment.trainX # the vocabulary dictionary and return term-document matrix

    #X is label
    #X = sentiment.trainX

    #X_arr = X.toarray()
    
    solve_name = 'sag'
    penalty = 'l2'
 
    cls = train_classifier(sentiment.trainX, sentiment.trainy, penalty = penalty, solver = solve_name)
    #stoplist = get_stopwords(datanum,sentiment.count_vect.get_feature_names(),cls.coef_[0])
    acc = evaluate(sentiment.devX, sentiment.devy, cls, 'dev data')

    #print('before using stopwords' )
    print('before using unlabeled data ,acc : {}'.format(acc))


    unlabeled = sentimentinterface.read_unlabeled(tarfname, sentiment)



   ##############   Train on unlabeled data ##################
    sentiment_now =sentiment
    unlabeled_num = unlabeled.X.shape[0]
    acc_dict = {}
    acc_dict2 = {}
    cls_pre = cls
    pre_labels=sentiment.devy
    data_num = unlabeled.X.shape[0]
    n =500
    used_num =0
    p =0
    k=1
    data_xaxis=[]
    data_yaxis=[]
    item_list = list(range(0, data_num, n))

    #for i in item_list:
        
    #for n in np.linspace(1000,20000,11):
        #if (i > 50000):

            #break
    #data_xaxis.append(i/data_num)
    #print(len(cls_pre.coef_[0]))
    unlabeled = sentimentinterface.read_unlabeled(tarfname, sentiment_now)
    batchdata_x = unlabeled.X[0:30000,:] 
    #batchdata_x = unlabeled.X[0:n,:] 
    #print(batchdata_x.shape)
    yprob = cls_pre.predict_proba(batchdata_x) 
    confidence_y = [max(g) for g in yprob]
    useful_idx = np.array(confidence_y) > 0.98
    #print(useful_idx)

    unlabeled_data1= np.array(unlabeled.data[0:30000])
    unlabeled_x = list(unlabeled_data1[useful_idx])
    unlabeled_xvec = unlabeled.X[0:30000,:]
    #unlabeled_x = np.array(unlabeled.data[0:n])
    #print(unlabeled_x)
    unlabeled_y = cls_pre.predict(unlabeled_xvec[useful_idx])

    batch_num = sum(useful_idx)
    used_num += batch_num
    #print(unlabeled_y)
    unlabeled_y=list(sentiment.le.inverse_transform(list(unlabeled_y)))
    #print(sentiment_now.trainX.shape[0],sentiment_now.trainy.shape[0])
    sentiment_now = sentimentinterface.read_combine_files(sentiment, sentiment_now, unlabeled_x, unlabeled_y, min_df = mindf, max_df = maxdf)
    #print(sentiment_now.trainX.shape[0],sentiment_now.trainy.shape[0])
    cls_now = train_classifier(sentiment_now.trainX, sentiment_now.trainy, penalty = penalty, solver = solve_name)
    acc = evaluate(sentiment_now.devX, sentiment_now.devy, cls_now, 'dev data')
    data_yaxis.append(acc)
    print('accuracy after combining {0} {1} {2} ({3:.3f}) unlabeled data: {4:.4f}'.format(batch_num, used_num, data_num,used_num / data_num, acc))  
    cls_pre = cls_now
    #acc_dict[i] = acc
    #if (i/data_num)>p*0.01:
    #    acc_dict[p*0.01]=acc
    #    p+=1
    #    print('used {} percent unlabeled data'.format(p))
    #    print(acc_dict)
        
    plt.figure()
    plt.plot(data_xaxis,data_yaxis)
    plt.title('Accuracy versus size of unlabeled data')
    plt.xlabel('Unlabeled data size (%)')
    plt.ylabel('Accuracy')
    plt.show()
        #k+=1
    print(finished)


    for j in range(400):
        review = sentiment.dev_data[j]
        dev_X = sentiment.count_vect.transform(reviews)
        yp = cls_now.predict_proba(dev_X) 

    posible_false= yp[0.4<yp<0.6]

            
    
        #print(review)
                



    #max_key = max(acc_dict.keys(), key=(lambda k:acc_dict[k]))
    #print(max_key)
    #print('Maximum accuracy happens at :{}, {}',format(max_key, str(acc_dict[max_key])))
    ###########################################
    








    #acc_dict[(mindf,maxdf,datanum,'before')]=acc
    #Read unlabeled data
    
    #sentiment = sentimentinterface.read_files(tarfname, stop_words = stoplist, min_df = mindf, max_df = maxdf)
    #array = sentiment.trainX # the vocabulary dictionary and return term-document matrix
    #X = sentiment.trainX
    
    #print(sentiment.count_vect.inverse_transform(X)[0]) # Return terms with nonzero entries in X[0].
    #X_arr = X.toarray()

    #cls = train_classifier(sentiment.trainX, sentiment.trainy, penalty = penalty, solver = solve_name)
    #acc = evaluate(sentiment.devX, sentiment.devy, cls, 'dev data')
    #print('after using stopwords' )
    #print('when using using min_df = {}, MAX_DF ={}, datanum ={}, acc : {}'.format(mindf, maxdf, datanum, acc))   
    #acc_dict[(mindf,maxdf,datanum,'after')]=acc   

    ########### Write Kaggle File ############
    #unlabeled = sentimentinterface.read_unlabeled(tarfname, sentiment)
    #sentimentinterface.write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
    #print('finished writing Kaggle file')

    ##########################################


    #print(acc_dict)  
    #print( max(acc_dict.items(), key=lambda k: k[1]))


            #print('when using solver :{}, penalty: {}, the accuracy is {}'.format(solve_name , penalty, acc))

            # It seems like using sag algorithm with l2 penalty could get best result.
'''
    solver= ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    #solver = ['lbfgs']
    solver_onlyl2= ['newton-cg', 'lbfgs', 'sag']
    solver_onlyl1= ['liblinear',  'saga']

    for penalty in ['l1','l2']:
    #for penalty in ['l2']:    

        for solve_name in solver:
            #if (penalty is 'l1' and solve_name not in solver_onlyl2) or (penalty is 'l2' and solve_name not in solver_onlyl1):
            if (penalty is 'l1' and solve_name not in solver_onlyl2) or (penalty is 'l2' ):

                cls = train_classifier(sentiment.trainX, sentiment.trainy, penalty = penalty, solver = solve_name)
                
                acc = evaluate(sentiment.devX, sentiment.devy, cls, 'dev data')
                
                print('when using solver :{}, penalty: {}, the accuracy is {}'.format(solve_name , penalty, acc))
'''

