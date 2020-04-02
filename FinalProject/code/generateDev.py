import utility
import sentiment
import tarfile

tar = tarfile.open('../data/sentiment.tar.gz', "r:gz")
devname = 'dev.tsv'
for member in tar.getmembers():
    if 'dev.tsv' in member.name:
        devname = member.name
print(devname)

data, labels = sentiment.read_tsv(tar, devname)
utility.getInitialization(200)

with open('devPred.txt', 'w') as fout, open('overConfident.txt', 'w') as fout2:
    for line, label in zip(data, labels):
        _, ov = utility.getNewInputVector([line])
        pred, prob = utility.getModelPrediction(ov)
        if pred[0] == 0:
            pred = 'NEGATIVE'
        elif pred[0] == 1:
            pred = 'POSITIVE'
        prob = max(prob[0])
        fout.write(pred + ' ' + line + ' '+ str(prob) + '\n')
        if pred != label and prob >= 0.9:
            fout2.write(pred + ' ' + label + ' ' + str(prob) + ' ' + line + '\n' )
print('done output devPred')


