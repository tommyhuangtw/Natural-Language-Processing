#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('EOS', sentence)
        return p
    
    # required, update the model when a sentence is observed

    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words



############____MyTrigram____###########################
class Trigram(LangModel):
    def __init__(self , thres = 4, backoff = 0.000001,smoothing='laplace'):
        self.model = dict()
        self.model_onestr=dict()
        #self.conmodel = dict()
        self.lbackoff = log(backoff, 2)
        self.thres=thres
        self.Vdelta=0.001
        self.smoothing = smoothing
        
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence1(s)
        new_corpus=self.change_corpus(corpus)
        self.model_onestr=dict()
        
        for s in new_corpus:
            self.fit_sentence2(s)
        #print(self.model.items())
        #print(self.model_onestr.items())
        self.norm()
        
    def fit_sentence1(self, sentence): 
    # optional, if there are any post-training steps (such as normalizing probabilities)
        sentence=['*','*']+sentence+['EOS']
        for idx in range(len(sentence)):
            self.inc_oneword(sentence[idx])
    
    def fit_sentence2(self, sentence): 
        sentence=['*','*']+sentence+['EOS']
        for idx in range(len(sentence)):
            self.inc_oneword(sentence[idx])
        for idx in range(len(sentence)-2):
            self.inc_words((sentence[idx:idx+3]))
        #self.inc_word('END_OF_SENTENCE')
        for idx in range(len(sentence)-1):
            self.inc_words((sentence[idx:idx+2]))
        
    def change_corpus(self,corpus):
        unk_list=[]
        new_corpus=corpus.copy()
        unk_list = [fewkey for fewkey in self.model_onestr.keys() if \
                    self.model_onestr[fewkey]<self.thres ]
    
        for sentence in new_corpus:
            for i in range(len(sentence)):
                if sentence[i] in unk_list:
                    sentence[i]='UNK'         
        return new_corpus
        
    def inc_words(self, w):
        if tuple(w) in self.model:
            self.model[tuple(w)] += 1.0
        else:
            self.model[tuple(w)] = 1.0
            
    def inc_oneword(self ,w):
        if w in self.model_onestr:
            self.model_onestr[w] += 1.0
        else:
            self.model_onestr[w] = 1.0
            
    def print_keys(self):
        print(self.model.keys())
        print(self.model_onestr.keys())
        
    def norm(self): pass

    def perplexity(self ,corpus,  Vdelta = 0.1):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        self.Vdelta = Vdelta
        return pow(2.0, self.entropy(corpus))
    
    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)
    
    def logprob_sentence(self, sentence):
        p = 0.0
        
        if self.smoothing == 'laplace':
            for i in xrange(len(sentence)):
                p += self.cond_logprob(sentence[i], sentence[:i])
            p += self.cond_logprob('EOS', sentence)
            return p
        if self.smoothing == 'interInterp':
            for i in xrange(len(sentence)):
                p += self.cond_logprob_interpol(sentence[i], sentence[:i])
            p += self.cond_logprob_interpol('EOS', sentence)
            return p
        
    def cond_logprob_interpol(self, word, previous):
        cond_pre2 = tuple(previous[-2:]+[word])
        cond_pre1 = tuple(previous[-1:]+[word])
        cond_pre0= word
        iter_list =[i * 0.25 for i in list(range(5))]
        cond_prob=0
        self.interpol_dict=dict()
        
        for lambda3 in iter_list:
            if cond_pre0 in self.model_onestr:
                conprob0 = self.model_onestr[cond_pre0] / self.wordlen()
                cond_prob += lambda3 * conprob0
            for lambda2 in iter_list:  
                
                if cond_pre1 in self.model:
                    conprob1 = self.model[cond_pre1] / self.model_onestr[cond_pre0]
                    cond_prob += lambda2 * conprob1
                lambda1 = 1-lambda2-lambda3
                
                if cond_pre2 in self.model:
                    conprob2 = self.model[cond_pre2]/self.model[cond_pre1]
                    cond_prob += lambda1 * conprob2
                    
                    # use log(con_Prob)
        return lconprob
            

    def cond_logprob(self, word, previous):
        cond_pre2 = tuple(previous[-2:]+[word])
        cond_pre1 = tuple(previous[-1:]+[word])
        tmp= self.Vdelta*self.wordlen()
        if cond_pre2 in self.model:
            conprob = (self.model[cond_pre2]+1)/(self.model[cond_pre1]+tmp)        
            # use log(con_Prob)
            lconprob=log(conprob,2)
            return lconprob
        else:
            return log((1/tmp),2)
            #return self.lbackoff
    
    def vocab(self):
        return self.model_onestr.keys()
    
    def wordlen(self):
        return len(self.model_onestr.keys())


######___________________Bigram____________________#############
class Bigram(LangModel):
    def __init__(self, backoff = 0.000001,smoothing='laplace'):
        self.model = dict()
        self.model_onestr=dict()
        #self.conmodel = dict()
        self.lbackoff = log(backoff, 2)
        self.thres=4
        self.Vdelta=0.001
        self.smoothing = smoothing
        
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence1(s)
        new_corpus=self.change_corpus(corpus)
        self.model_onestr=dict()
        
        for s in new_corpus:
            self.fit_sentence2(s)
        #print(self.model.items())
        #print(self.model_onestr.items())
        self.norm()
        
    def fit_sentence1(self, sentence): 
    # optional, if there are any post-training steps (such as normalizing probabilities)
        sentence=['*','*']+sentence+['EOS']
        for idx in range(len(sentence)):
            self.inc_oneword(sentence[idx])
    
    def fit_sentence2(self, sentence): 
        sentence=['*','*']+sentence+['EOS']
        for idx in range(len(sentence)):
            self.inc_oneword(sentence[idx])
        #self.inc_word('END_OF_SENTENCE')
        for idx in range(len(sentence)-1):
            self.inc_words((sentence[idx:idx+2]))
        
    def change_corpus(self,corpus):
        unk_list=[]
        new_corpus=corpus.copy()
        unk_list = [fewkey for fewkey in self.model_onestr.keys() if \
                    self.model_onestr[fewkey]<self.thres ]
    
        for sentence in new_corpus:
            for i in range(len(sentence)):
                if sentence[i] in unk_list:
                    sentence[i]='UNK'         
        return new_corpus
        
    def inc_words(self, w):
        if tuple(w) in self.model:
            self.model[tuple(w)] += 1.0
        else:
            self.model[tuple(w)] = 1.0
            
    def inc_oneword(self ,w):
        if w in self.model_onestr:
            self.model_onestr[w] += 1.0
        else:
            self.model_onestr[w] = 1.0
            
    def print_keys(self):
        print(self.model.keys())
        print(self.model_onestr.keys())
        
    def norm(self): pass

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))
    
    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)
    
    def logprob_sentence(self, sentence):
        p = 0.0
        
        if self.smoothing == 'laplace':
            for i in xrange(len(sentence)):
                p += self.cond_logprob(sentence[i], sentence[:i])
            p += self.cond_logprob('EOS', sentence)
            return p

    def cond_logprob(self, word, previous):
        cond_pre1 = tuple(previous[-1:]+[word])
        tmp= self.Vdelta*self.wordlen()
        if cond_pre1 in self.model:
            conprob = (self.model[cond_pre1]+1)/(self.model_onestr[word]+tmp)        
            # use log(con_Prob)
            lconprob=log(conprob,2)
            return lconprob
        else:
            return log((1/tmp),2)
            #return self.lbackoff
    
    def vocab(self):
        return self.model_onestr.keys()
    
    def wordlen(self):
        return len(self.model_onestr.keys())


class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()

