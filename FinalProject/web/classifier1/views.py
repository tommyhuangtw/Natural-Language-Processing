from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
import sys, os

from sklearn.manifold import MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np

sys.path.insert(1, os.path.join(os.path.realpath(os.path.pardir), 'code'))
from utility import utility

# mds = MDS(n_components = 2)
lda = LDA()
allVectors = np.array([])
util = utility('../code/pos_neg_matrix.pkl','../code/model.pkl', '../code/vectorizer.pkl')
k = 50
data1 = util.getInitialization(k)
n = 300
allVectors = np.concatenate((np.random.permutation(data1['positiveCorpusVectors'])[:n], np.random.permutation(data1['negativeCorpusVectors'])[:n]))
data3 = util.getConfidence(allVectors)
allVectors = allVectors[:, data1['featureIndex']]
lda.fit(allVectors, [0] * n + [1] * n)

def index(request):
    return render(request, 'classifier1/index.html')

@csrf_exempt
def predict(request):
    global allVectors, util, lda
    
    data = util.getSimilarityInfo([request.POST['sentence']])

    # corpusVectors = mds.fit_transform(np.concatenate((allVectors, [data['sentenceVector']])))
    # data['sentenceVector'] = corpusVectors[-1].tolist()
    data['sentenceVector'] = lda.transform([data['sentenceVector']]).tolist() + [data['confidence']]

    data['sentence'] = request.POST['sentence']
    data['confidence'] = round(data['confidence'], 3) * 100
    data['positiveCosineSimilarity'] = round(data['positiveCosineSimilarity'], 3)
    data['negativeCosineSimilarity'] = round(data['negativeCosineSimilarity'], 3)
    data['positiveOccurrences'] = ', '.join(data['positiveOccurrences'])
    data['negativeOccurrences'] = ', '.join(data['negativeOccurrences'])
    response = JsonResponse(data)
    return response

@csrf_exempt
def initialize(request):
    global allVectors, n, data3, lda, data1, util
    data2 = util.getSimilarityInfo([request.POST['sentence']])
    data = dict(data1, **data2)

    # corpusVectors = mds.fit_transform(np.concatenate((allVectors, [data['sentenceVector']])))
    # data['positiveCorpusVectors'] = corpusVectors[:n]
    # data['negativeCorpusVectors'] = corpusVectors[n:]
    data['positiveCorpusVectors'] = np.concatenate((lda.transform(allVectors[:n]), data3[:n]), axis = 1).tolist()
    data['negativeCorpusVectors'] = np.concatenate((lda.transform(allVectors[n:]), data3[n:]), axis = 1).tolist()

    data['sentenceVector'] = lda.transform([data['sentenceVector']]).tolist() + [data['confidence']]

    data['confidence'] = round(data['confidence'], 3) * 100
    data['sentence'] = request.POST['sentence']
    data['positiveCosineSimilarity'] = round(data['positiveCosineSimilarity'], 3)
    data['negativeCosineSimilarity'] = round(data['negativeCosineSimilarity'], 3)
    data['positiveOccurrences'] = ', '.join(data['positiveOccurrences'])
    data['negativeOccurrences'] = ', '.join(data['negativeOccurrences'])
    # data = {
    #     'sentence': request.POST['sentence'],
    #     'result': 'POSITIVE',
    #     'sentenceVector': [],
    #     'positiveCosineSimilarity': 0.8,
    #     'negativeCosineSimilarity': 0.2,
    #     'positiveOccurrences': ', '.join(list(['a', 'b'])),
    #     'negativeOccurrences': ', '.join(list(['1', '2'])),

    #     'positiveCorpusVectors': [[]],
    #     'negativeCorpusVectors': [[]],
    #     'topWords': { 'foo': 12, 'bar': 6 },
    #     'bottomWords': { 'boo': 12, 'far': 6 },
    # }
    return render(request, 'classifier1/result.html', data)