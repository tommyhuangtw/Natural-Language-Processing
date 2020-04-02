import pickle
import numpy as np
import dill

class utility:
    def __init__(self, matrix_pkl, model_pkl, vectorizer_pkl):
        self.posVec = None
        self.negVec = None
        self.featureName = None
        self.modelCoef = None
        self.posFeature = None
        self.negFeature = None
        self.topKPosAvgVec = None
        self.topKNegAvgVec = None
        self.posMatrix = None
        self.negMatrix = None
        self.tfidfVec = None
        self.featureIndex = None
        self.model = None
        self.setUp(matrix_pkl, model_pkl, vectorizer_pkl)

    def setUp(self, fvec, fmodel, vectorizer):
        self.posVec, self.negVec, self.featureName = self.getVector(fvec)
        self.modelCoef = self.getModelWeight(fmodel)[0]
        with open(vectorizer, 'rb') as vin:
            self.tfidfVec = dill.load(vin).tfidf_vect


    def getVector(self, fname):
        with open(fname, 'rb') as fin:
            [pos, neg, feature_name] = pickle.load(fin)
        self.posMatrix = pos.toarray()
        self.negMatrix = neg.toarray()
        return pos.toarray().mean(axis=0), neg.toarray().mean(axis=0), feature_name

    def getModelWeight(self, fname):
        with open(fname, 'rb') as fin:
            self.model = pickle.load(fin)
        coef = self.model.coef_.tolist()
        return coef

    def getFeatureNameByIndex(self, index):
        ##### DEGUGGING #####
        print("index: {}",index)
        return [self.featureName[i] for i in index]


    def getNewInputVector(self, input_text):
        inputVec = self.tfidfVec.transform(input_text)
        return inputVec[:, self.featureIndex], inputVec


    def getCosineSimilarity(self, input_vec):
        '''get cosineSimilarity put in dict'''
        topKPosAvgVecReshaped = self.topKPosAvgVec.reshape((1, self.topKPosAvgVec.shape[0]))
        topKNegAvgVecReshaped = self.topKNegAvgVec.reshape((1, self.topKNegAvgVec.shape[0]))
        from sklearn.metrics.pairwise import cosine_similarity
        pos_sim = cosine_similarity(input_vec, topKPosAvgVecReshaped)
        neg_sim = cosine_similarity(input_vec,topKNegAvgVecReshaped)
        return pos_sim[0], neg_sim[0]


    def getKFeatures(self, k):
        '''
        Usage:
            input: k (top k + bottom k features)
            return: a length of 2k vector

        get top K features and last k features and concatenate to a feature vector of size 2k, featureWeight, featureIndex
        '''
        sortedWeightVec = sorted(self.modelCoef, reverse=True)
        sortedIndexVec = sorted(range(len(self.modelCoef)), key= lambda i: self.modelCoef[i], reverse=True)
        topFeatureWeight = sortedWeightVec[k:]
        bottomFeatureWeight = sortedWeightVec[-k:]
        topFeatureIndex = sortedIndexVec[:k]
        bottomFeatureIndex = sortedIndexVec[-k:]
        self.featureIndex  = topFeatureIndex + bottomFeatureIndex
        featureWeight = topFeatureWeight + bottomFeatureWeight
        self.posFeature = self.getFeatureNameByIndex(topFeatureIndex)
        self.negFeature = self.getFeatureNameByIndex(bottomFeatureIndex)
        # get topK features from posVec and negVec
        self.topKPosAvgVec = np.array([self.posVec[i] for i in self.featureIndex])
        self.topKNegAvgVec = np.array([self.negVec[i] for i in self.featureIndex])
        return featureWeight

    def getInitialization(self, k):
        '''
        output:
            n_pos*(topKfeatureVec)
            n_neg*(topKfeatureVec)
            topKWordDict (word: modelWeight)
            bottomWordDict (word: modelWeight)
        '''
        featWeight = self.getKFeatures(k)
        output = {}
        output['positiveCorpusVectors'] = self.posMatrix
        output['negativeCorpusVectors'] = self.negMatrix
        output['featureIndex'] = self.featureIndex
        topIndex = self.featureIndex[:int(len(self.featureIndex)/2)]
        bottomIndex = self.featureIndex[int(len(self.featureIndex)/2):]
        topKName = self.getFeatureNameByIndex(topIndex)
        bottomName = self.getFeatureNameByIndex(bottomIndex)
        topWordList, bottomWordList = {}, {}
        for i in range(len(topIndex)):
            topWordList[topKName[i]] = featWeight[topIndex[i]]
            bottomWordList[bottomName[i]] = featWeight[bottomIndex[i]]
        output['topWords'] = topWordList
        output['bottomWords'] = bottomWordList
        return output

    def getOccurences(self, input_sent):
        sent = input_sent[0].split(' ')
        posOccur, negOccur = [], []
        preWord = '_SOS_'
        for word in sent:
            if word in self.posFeature and word not in posOccur:
                posOccur.append(word)
            elif word in self.negFeature  and word not in negOccur:
                negOccur.append(word)
            bigram = preWord + ' ' + word
            if bigram in self.posFeature and bigram not in posOccur:
                posOccur.append(bigram)
            elif bigram in self.negFeature and bigram not in negOccur:
                negOccur.append(bigram)
            preWord = word
        return posOccur, negOccur


    def getModelPrediction(self, input_vect):
        y_hat = self.model.predict(input_vect)
        prob = self.model.predict_proba(input_vect)
        return y_hat, prob

    def getSimilarityInfo(self, inputSent, isToxic = False):
        output = {}
        inputVec, originalVec = self.getNewInputVector(inputSent)
        pos, neg = self.getCosineSimilarity(inputVec)
        posOccur, negOccur = self.getOccurences(inputSent)
        pred, prob = self.getModelPrediction(originalVec)
        if isToxic:
            if pred[0] == 0:
                output['result'] = 'NON-TOXIC👼'
            else:
                output['result'] = 'TOXIC☠️'
        else:
            if pred[0] == 0:
                output['result'] = 'NEGATIVE👎'
            else:
                output['result'] = 'POSITIVE👍'
        output['confidence'] = max(prob[0])
        output['sentenceVector'] = inputVec.toarray().tolist()[0]
        output['positiveCosineSimilarity'] = pos.tolist()[0]
        output['negativeCosineSimilarity'] = neg.tolist()[0]
        output['positiveOccurrences'] = posOccur
        output['negativeOccurrences'] = negOccur
        return output

    def getConfidence(self, vectorList):
        pred, prob = self.getModelPrediction(vectorList)
        return [ [max(p)] for p in prob ]

    def pcaForVisualization():
        pass

if __name__ == "__main__":
    ''' check getInitialzation '''
    dataUtil = utility('../code/pos_neg_matrix.pkl', '../code/model.pkl', '../code/vectorizer.pkl')
    k = 10
    output = dataUtil.getInitialization(k)
    assert output != None, "initialization Error"

    ''' check getNewInputVector'''
    inputSent = ["I would like to revisit this place again since this place serves great meal. Overall is awesome. "]
    similarityInfo = dataUtil.getSimilarityInfo(inputSent)
    print("prediction: ", similarityInfo['result'])
    print("sentenceVector: ", similarityInfo['sentenceVector'])
    print("positiveCosineSim: ",similarityInfo['positiveCosineSimilarity'])
    print("negCosineSim: ", similarityInfo['negativeCosineSimilarity'])
    print("confidence: ", similarityInfo['confidence'])
