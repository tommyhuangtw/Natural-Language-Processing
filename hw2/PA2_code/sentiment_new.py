#!/bin/python
from classify import train_classifier, evaluate
import numpy as np
def read_files(tarfname,min_df = 1, max_df =1.0, max_features= None,stop_words= None,ngram_range =(1,2)):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    #print("-- train data")
    #read train_data and train_labels and print out the length
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    #print(len(sentiment.train_data))

    #print("-- dev data")
    #read dev_data and dev_labels and print out the length
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    #print(len(sentiment.dev_data))
    #print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer

    
    #sentiment.count_vect = CountVectorizer(stop_words ='english', min_df = min_df,max_df = max_df, ngram_range=(1,2), max_features = max_features)
    sentiment.count_vect = CountVectorizer(stop_words = stop_words, min_df = min_df,max_df = max_df, ngram_range = ngram_range, max_features = max_features)
    #convert data into token and count frequency
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    #encode labels (le)
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    #le.classes_ = set of label names
    sentiment.target_labels = sentiment.le.classes_
    #Transform Categories Into Integers
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def read_combine_files(sentiment_pre, sentiment ,unlabeled_x, unlabeled_y, min_df = 1, max_df =1.0, max_features= None,stop_words= None):
    """Read old data and combine with unlabeled data
        """
    #tar = tarfile.open(tarfname, "r:gz")
    #devname = "dev.tsv"
    #sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    #sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    #print('len of train_data(before)',len(sentiment.train_data))
    sentiment.train_data += unlabeled_x
    #print('len of train_data',len(sentiment.train_data))
    #print('len of train_labels(before)',len(sentiment.train_labels))
    sentiment.train_labels += unlabeled_y
    #print('len of train_labels',len(sentiment.train_labels))    #print(len(sentiment.train_data))
    
    #print("-- dev data")
    #read dev_data and dev_labels and print out the length
    #sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    #print(len(sentiment.dev_data))
    #print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    
    #sentiment.count_vect = CountVectorizer(stop_words ='english', min_df = min_df,max_df = max_df, ngram_range=(1,2), max_features = max_features)
    sentiment.count_vect = CountVectorizer(stop_words = stop_words, min_df = min_df,max_df = max_df, ngram_range=(1,2), max_features = max_features)
    #convert data into token and count frequency
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment_pre.dev_data)
    from sklearn import preprocessing
    #encode labels (le)
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    #le.classes_ = set of label names
    sentiment.target_labels = sentiment.le.classes_
    #Transform Categories Into Integers
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment_pre.dev_labels)
    #tar.close()
    return sentiment

def read_combine_files_1(sentiment ,unlabeled_x, unlabeled_y, min_df = 1, max_df =1.0, max_features= None,stop_words= None):
    """Read old data and combine with unlabeled data
        """
    #import tarfile
    #tar = tarfile.open(tarfname, "r:gz")
    #devname = "dev.tsv"
    #sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    #sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print('len of train_data(before)',len(sentiment.train_data))
    sentiment.train_data += unlabeled_x
    print('len of train_data',len(sentiment.train_data))
    print('len of train_labels(before)',len(sentiment.train_labels))
    sentiment.train_labels += unlabeled_y
    print('len of train_labels',len(sentiment.train_labels))    #print(len(sentiment.train_data))
    
    #print("-- dev data")
    #read dev_data and dev_labels and print out the length
    #sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    #print(len(sentiment.dev_data))
    #print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    
    #sentiment.count_vect = CountVectorizer(stop_words ='english', min_df = min_df,max_df = max_df, ngram_range=(1,2), max_features = max_features)
    sentiment.count_vect = CountVectorizer(stop_words = stop_words, min_df = min_df,max_df = max_df, ngram_range=(1,2), max_features = max_features)
    #convert data into token and count frequency
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    #encode labels (le)
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    #le.classes_ = set of label names
    sentiment.target_labels = sentiment.le.classes_
    #Transform Categories Into Integers
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    #tar.close()
    return sentiment



def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
            #print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
#print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    #print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

if __name__ == "__main__":
    tarfname = "data/sentiment.tar.gz"
    maxdf = 1.0
    mindf = 1
    solve_name = 'sag'
    penalty = 'l2'

    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname, min_df = mindf, max_df = maxdf)
    print("\nTraining classifier")
    import classify
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
    print("\nEvaluating")
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    
    #probability =[0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.98]
    #for p in probability:
        
    cls= train_classifier(sentiment.trainX, sentiment.trainy, penalty = penalty, solver = solve_name)

    acc = evaluate(sentiment.devX, sentiment.devy, cls, 'dev data')
    print('when using using min_df = {}, MAX_DF ={},acc : {}'.format(mindf, maxdf, acc))
    
    n=1000
    #unlabeled = sentimentinterface.read_unlabeled(tarfname, sentiment)
    
    #sentiment = read_files(tarfname, min_df = mindf, max_df = maxdf)
    
    unlabeled_x = unlabeled.data
    unlabeled_x = np.array(unlabeled_x)
    unlabeled_y = cls.predict(unlabeled.X)
    
    unlabeled_y=sentiment.le.inverse_transform(list(unlabeled_y))
    batchdata_x = unlabeled.X
    yprob = cls.predict_proba(batchdata_x)
    confidence_y = [max(g) for g in yprob]
    useful_idx = np.array(confidence_y) > 0.98
    unlabeled_x =list(unlabeled_x[useful_idx])
    unlabeled_y =list (unlabeled_y[useful_idx])
    
    sentiment1 = read_combine_files_1(sentiment ,unlabeled_x, unlabeled_y)

    cls_now = train_classifier(sentiment1.trainX, sentiment1.trainy, penalty = penalty, solver = solve_name)
    acc = evaluate(sentiment1.devX, sentiment1.devy, cls_now, 'dev data')
    print('accuracy after probability {} unlabeled data: {}'.format(0.98, acc))
    
    
    coef_pre = cls.coef_[0]
    print('shape of coef_pre:' ,coef_pre.shape)
    
    coef_now =cls_now.coef_[0]
    print('shape of coef_now: ',coef_now.shape)
    
    diff_dict={}
    X = sentiment.trainX
    X1 = sentiment1.trainX
    w1 = sentiment.count_vect.get_feature_names()
    w2 = sentiment1.count_vect.get_feature_names()
    print('ready to calculate the diff')
    print('len of w1',len(w1))
    print('len of w2',len(w2))
    print(w1[0:10])
    for i in w1:
        if i in w2:
            idx1 =w1.index(i)
               #print(idx1)
            idx2 =w2.index(i)
           #print(idx2)
            diff_dict[i]=abs(coef_pre[idx1]-coef_now[idx2])

    k=20
    top_k =np.argsort(np.array(diff_dict.values()))[-k:]
    print('most interesting features')
    for i in top_k:
        
        # use sentiment.count_vect.get_feature_names()[i] to get word name by index
        print(sentiment.count_vect.get_feature_names()[i])
        top_k_words.append(sentiment.count_vect.get_feature_names()[i])
    print('finished')
    print(top_k_words)
#diff = np.abs(coef_pre-coef_now)
#print(deff[0:3])
#sort_coef_diff =np.argsort(coefficients)
    
#top_k= sort_coef_diff[-10:]
#print(top_k)
#print((sentiment.count_vect.get_feature_names()[top_k]))
    #print("Writing predictions to a file")
    #write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
    #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
