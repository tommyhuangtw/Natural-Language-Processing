import pickle
import numpy as np

posVec = None
negVec = None
featureName = None
modelCoef = None
posFeature = None
negFeature = None
topKPosAvgVec = None
topKNegAvgVec = None
posMatrix = None
negMatrix = None
tfidfVec = None
featureIndex = None
model = None

def setUp(fvec, fmodel, vectorizer):
    '''
        fvec: [pos, neg, feature_name]
        ([sentiment.pos_matrix, sentiment.neg_matrix, sentiment.tfidf_vect.get_feature_names()])
        fmodel: cls
        vectorizer: sentiment
    '''
    
    print("Setting up fvec, fmodel, vectorizer")
    import dill
    global posVec, negVec, featureName, modelCoef, tfidfVec
    posVec, negVec, featureName = getVector(fvec)
    print("finished getting fvec")
    modelCoef = getModelWeight(fmodel)[0]
    print("reading Vectorizer: ", vectorizer)
    with open(vectorizer, 'rb') as vin:
        tfidfVec = dill.load(vin).tfidf_vect

def getVector(fname):
    global posMatrix, negMatrix
    print("loading fvec:", fname)
    fin = open(fname, 'rb')
    [pos, neg, feature_name] = pickle.load(fin)
    print("shape of pos:", pos.shape)
    print("shape of neg:", neg.shape)
    print("length of feature name :", len(feature_name))
    
    fin.close()
    print('processing pos array')
    posMatrix = pos.toarray()
    print('processing neg array')
    negMatrix = neg.toarray()
    return pos.toarray().mean(axis=0), neg.toarray().mean(axis=0), feature_name

def getModelWeight(fname):
    print("geting model weight: ", fname)
    fin = open(fname, 'rb')
    global model
    model = pickle.load(fin)
    coef = model.coef_.tolist()
    #print(coef.index(max(coef))
    fin.close()
    return coef

def getFeatureNameByIndex(index):
    return [featureName[i] for i in index]

def getNewInputVector(input_text):
    global tfidfVec, featureIndex
    inputVec = tfidfVec.transform(input_text)
    return inputVec[:, featureIndex], inputVec

def getCosineSimilarity(input_vec):
    '''get cosineSimilarity put in dict'''
    global topKPosAvgVec, topKNegAvgVec

    topKPosAvgVecReshaped = topKPosAvgVec.reshape((1, topKPosAvgVec.shape[0]))
    topKNegAvgVecReshaped = topKNegAvgVec.reshape((1, topKNegAvgVec.shape[0]))
    from sklearn.metrics.pairwise import cosine_similarity
    pos_sim = cosine_similarity(input_vec, topKPosAvgVecReshaped)
    neg_sim = cosine_similarity(input_vec,topKNegAvgVecReshaped)

    return pos_sim[0], neg_sim[0]

def getKFeatures(k):
    '''
    Usage:
        input: k (top k + bottom k features)
        return: a length of 2k vector

    get top K features and last k features and concatenate to a feature vector of size 2k, featureWeight, featureIndex
    '''

    global featureName, modelCoef, posFeature, negFeature, topKPosAvgVec, topKNegAvgVec, featureIndex
    sortedWeightVec = sorted(modelCoef, reverse=True)
    sortedIndexVec = sorted(range(len(modelCoef)), key= lambda i: modelCoef[i], reverse=True)
    topFeatureWeight = sortedWeightVec[k:]
    bottomFeatureWeight = sortedWeightVec[-k:]
    topFeatureIndex = sortedIndexVec[:k]
    bottomFeatureIndex = sortedIndexVec[-k:]
    featureIndex  = topFeatureIndex + bottomFeatureIndex
    featureWeight = topFeatureWeight + bottomFeatureWeight
    posFeature = getFeatureNameByIndex(topFeatureIndex)
    negFeature = getFeatureNameByIndex(bottomFeatureIndex)
    # get topK features from posVec and negVec
    topKPosAvgVec = np.array([posVec[i] for i in featureIndex])
    topKNegAvgVec = np.array([negVec[i] for i in featureIndex])

    return featureWeight

def getInitialization(k):
    '''
    output:
        n_pos*(topKfeatureVec)
        n_neg*(topKfeatureVec)
        topKWordDict (word: modelWeight)
        bottomWordDict (word: modelWeight)
    '''
    global posMatrix, negMatrix, posFeature, negFeature, featureIndex
    setUp('pos_neg_matrix_toxic.pkl','model_toxic.pkl','vectorizer_toxic.pkl')
    #setUp('pos_neg_matrix.pkl','model.pkl','vectorizer.pkl')
    if posFeature == None or negFeature == None:
        featWeight = getKFeatures(k)
    output = {}
    output['positiveCorpusVectors'] = posMatrix[:,featureIndex]
    output['negativeCorpusVectors'] = negMatrix[:,featureIndex]
    topIndex = featureIndex[:int(len(featureIndex)/2)]
    bottomIndex = featureIndex[int(len(featureIndex)/2):]
    topKName = getFeatureNameByIndex(topIndex)
    bottomName = getFeatureNameByIndex(bottomIndex)
    topWordList, bottomWordList = {}, {}

    for i in range(len(topIndex)):
        topWordList[topKName[i]] = featWeight[topIndex[i]]
        bottomWordList[bottomName[i]] = featWeight[bottomIndex[i]]

    output['topWords'] = topWordList
    output['bottomWords'] = bottomWordList
    return output

def getOccurences(input_sent):
    sent = input_sent[0].split(' ')
    posOccur, negOccur = [], []
    for word in sent:
        if word in posFeature:
            posOccur.append(word)
        elif word in negFeature:
            negOccur.append(word)
    return posOccur, negOccur

def getModelPrediction(input_vect):
    global model
    y_hat = model.predict(input_vect)
    prob = model.predict_proba(input_vect)
    return y_hat, prob[0]

def getSimilarityInfo(inputSent):
    output = {}
    inputVec, originalVec = getNewInputVector(inputSent)
    pos, neg = getCosineSimilarity(inputVec)
    posOccur, negOccur = getOccurences(inputSent)
    pred, prob = getModelPrediction(originalVec)
    if pred[0] == 0:
        output['result'] = 'NON-TOXIC'
    else:
        output['result'] = 'TOXIC'
    output['confidence'] = max(prob)
    output['sentenceVector'] = inputVec.toarray().tolist()[0]
    output['positiveCosineSimilarity'] = pos.tolist()[0]
    output['negativeCosineSimilarity'] = neg.tolist()[0]
    output['positiveOccurrences'] = posOccur
    output['negativeOccurrences'] = negOccur
    return output


def pcaForVisualization():
    pass

if __name__ == "__main__":
    ''' check getInitialzation '''
    k = 20
    output = getInitialization(k)
    assert output != None, "initialization Error"

    ''' check getNewInputVector'''
    inputSent = ["I'll happily dance on Wikipedia's grave when the shutdown happens, what a shit site that supports retards and vandals."]

    similarityInfo = getSimilarityInfo(inputSent)
    print("Input Sentence: ", inputSent)
    print("prediction: ", similarityInfo['result'])
    print("sentenceVector: ", similarityInfo['sentenceVector'])
    print("positiveCosineSim: ",similarityInfo['positiveCosineSimilarity'])
    print("negCosineSim: ", similarityInfo['negativeCosineSimilarity'])
    print("confidence: ", similarityInfo['confidence'])

    inputSent = ["Welcome to wikipedia. Thank you for experimenting and we hope you like the place and are looking forward to your contributions!"]
    similarityInfo = getSimilarityInfo(inputSent)
    print("Input Sentence: ", inputSent)
    print("prediction: ", similarityInfo['result'])
    print("sentenceVector: ", similarityInfo['sentenceVector'])
    print("positiveCosineSim: ",similarityInfo['positiveCosineSimilarity'])
    print("negCosineSim: ", similarityInfo['negativeCosineSimilarity'])
    print("confidence: ", similarityInfo['confidence'])
