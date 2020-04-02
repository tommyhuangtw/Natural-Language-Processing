#! /usr/bin/python

import sys
from collections import defaultdict
import math
import numpy as np
from string import punctuation

"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""
def rare_classifier1(word):
    '''
    Classify rare words into informative class
    '''
    newword = '_RARE_'

    return newword

def rare_classifier1(word):
    '''
    Classify rare words into informative class
    '''
    if word.isdigit():
        newword = "_NUMBER_"
    elif len(word)>10:
        newword = "_LongWORDs_"
    else:
        newword = '_RARE_'

    return newword

def rare_classifier1(word):
    '''
    Classify rare words into informative class
    '''
    if word.isdigit():
        newword = "_NUMBER_"
    elif len(word)>0 and word[0].isupper():
        newword = "_HeadOfSentence_"
    elif any((w in punctuation) for w in word):
        newword = "_PUNCTUATION_"
    elif len(word)>10:
        newword = "_LongWORDs_"
    elif (',' in word or '.' in word):
        newword = "_ContainDOT_"

    else:
        newword = '_RARE_'
    
    return newword

#####  fewer class  ######
def rare_classifier1(word):
    '''
    Classify rare words into informative class
    '''
    if (any(char.isdigit() for char in word)):
        newword = "_ONENUM_"
    elif word.isupper():
        newword = "_CAPITAL_"
    elif len(word)>10:
        newword = "_LongWORDs_"
    elif len(word)>0 and word[0].isupper():
        newword = "_HeadOfSentence_"
    elif (',' in word or '.' in word):
        newword = "_ContainDOT_"
    else:
        newword = '_RARE_'
    return newword
def rare_classifier1(word):
    '''
    Classify rare words into informative class
    '''
    if word.isdigit():
        newword = "_NUMBER_"
    elif len(word)>10:
        newword = "_LongWORDs_"
    else:
        newword = '_RARE_'
    return newword

def rare_classifier(word):
    '''
    Classify rare words into informative class
    '''
    if word.isdigit():
        newword = "_NUMBER_"
    elif (any(char.isdigit() for char in word)):
        newword = "_ONENUM_"
    elif len(word) == 1:
        newword = "_ONE_"
    elif word.isupper():
        newword = "_CAPITAL_"
    elif len(word)>0 and word[0].isupper():
        newword = "_HeadOfSentence_"
    elif (',' in word or '.' in word):
        newword = "_ContainDOT_"
    elif any((w in punctuation) for w in word):
        newword = "_PUNCTUATION_"
    elif len(word)>10:
        newword = "_LongWORDs_"
    else:
        newword = '_RARE_'
    #print(word, newword)
    return newword

def simple_conll_corpus_iterator(corpus_file, rerun = 'False', rare_list = None):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    i=1
    while l:
        #print('l is: {}'.format(l))
        line = l.strip()
        #if i ==300: break
        if line: # Nonempty line
            #print('line is : '.format(line))
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            #print('fields are : {}'.format(fields))
            ne_tag = fields[-1]

            word = " ".join(fields[:-1])

            if (rare_list is not None) and (word in rare_list):
                #word = rare_classifier(word)
                word = "_RARE_"
            else: 
                word = " ".join(fields[:-1])

            yield word, ne_tag
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()
        i+=1
   
def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                #print('current sentence : '.format(current_sentence))
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in range(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        

class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """
    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.emission_freq=defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()
        self.tagger_dict = defaultdict(lambda: 'I-GENE') 
        self.ngram_freq =defaultdict(int)
        self.rare_list = []
        

    def train(self, corpus_file, rerun = 'False', rare_list = None ):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        if rerun is 'True':
            print("Start Retraining with rare words")

        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file, rerun, rare_list)), self.n,)
        # get_ngrams: generate a tuple of ngram by sentence_iterator
        # sentence_iterator: generate sentence as lists of (word, ne_tag) tuples.
        # corpus_iterator : generate typle of (word, ne_tag)
        for ngram in ngram_iterator:
            #ngram is a list of tuple (word, ne_tag)
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags
            # tagson : sequence of tags
            for i in range(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def train_viterbi(self, input_counts, dev_file,output):
        '''
        Use Viterbi Algorithm to retrieve tag
        '''
        self.read_counts(input_counts)
        self.compute_emission()
        self.compute_ngram_freq()
        l = dev_file.readline()
        word_corpus = set([x[0] for x in self.emission_counts.keys()])
        #print("length of word_corpus:", len(word_corpus))
        c = 0
        print('CHECK _RARE_ VALUE')

        print(self.emission_freq[('_RARE_',"O")])
        print(self.emission_freq[('_RARE_',"I-GENE")])

        while l:
            c += 1
            #print('processing the {}th sentence'.format(c))
            #if c == 5: break
            sentence=[]
            while l is not '\n' :
                #print('End of a sentence')
                sentence.append(l[:-1])
                l=dev_file.readline()

            n=len(sentence)
            y_tag = [None] * n
            all_state = list(set([key[1] for key in list(self.emission_counts.keys())]))

            Sv = all_state

            pi_prob= defaultdict()
            pi_prob[(0,'*','*')]=1
            bp = [None] * n
            #print("sentence length: ", n)
            #print("sentence: ",sentence)
            for w_idx in range(1,n+1):
                #print('now is processing the {}th word: {}'.format(w_idx,sentence[w_idx-1]))
                bp[w_idx-1]= dict()

                if w_idx< 2 : Su = ['*']
                else : Su = all_state 

                if w_idx < 3: Sw = ['*']
                else: Sw = all_state 
                #print("length of rare_list:")
                #print(len(self.rare_list))
                
                # Determine the rare word    
                #word_used = sentence[w_idx-1]
                if sentence[w_idx-1] not in word_corpus:
                    #print('now is using rare word')
                    word_used = rare_classifier(sentence[w_idx-1])
                    #word_used = '_RARE_'
                else :
                    word_used = sentence[w_idx-1]
                #print("Word_used: ",word_used)
                for u in range(len(Su)):
                    # print("Su is :",Su[u])
                    for v in range(len(Sv)):
                        pi_tmp = []
                        
                        for w in range(len(Sw)):

                            tmp = pi_prob[(w_idx-1,Sw[w],Su[u])] * \
                            self.ngram_freq[(Sv[v],Sw[w],Su[u])] * \
                            self.emission_freq[(word_used,Sv[v])]
                            pi_tmp.append(tmp)


                        pi_prob[(w_idx,Su[u],Sv[v])] = np.max(pi_tmp)
                        #print("PI_PROB: ",pi_prob[(w_idx,Su[u],Sv[v])])
                        #print("piprob: ", pi_prob[(w_idx,Su[u],Sv[v])])
                        bp[w_idx-1][(Su[u],Sv[v])] = Sw[np.argmax(pi_tmp)]
                        #print("BP: ", bp[w_idx-1][(Su[u],Sv[v])] )

            #print('finised finding max prob per word')
            Su = all_state
            Sv = all_state 
            prob_tmp = np.zeros((len(Su),len(Sv)), dtype = np.float32)
            #print('finding the tag of last two words')
            
            for u in range(len(Su)):
                for v in range(len(Sv)):
                    prob_tmp[u,v] = pi_prob[(n,Su[u],Sv[v])]* self.ngram_freq[('STOP',Su[u],Sv[v])]
            #print("prob_temp : ",prob_tmp)
            idx_u,idx_v = np.unravel_index(prob_tmp.argmax(), prob_tmp.shape)

            #print("idx_u = {}, idx_v = {} ".format(idx_u, idx_v))
            y_tag[n-2], y_tag[n-1] = Su[idx_u], Sv[idx_v]

            for k in range(n-3,-1,-1):
                y_tag[k] = bp[k+2][(y_tag[k+1],y_tag[k+2])]

            #for i in range(n):
            #   print('{} : {}'.format(sentence[i],y_tag[i]))        

            # Write into evaluation file
            for i, word in enumerate(sentence):
                output.write("%s %s\n" % (word,y_tag[i]))
            # Read '\n' to process next sentence
            l = dev_file.readline()
            output.write("\n")
            #exit()

    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        print('Start writing to gene.counts')
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:            
            if word == "_RARE_":
                print(self.emission_counts[(word, ne_tag)])
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))

        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))
        print("finished writing to gene.counts")

    def read_counts(self, corpusfile):
        print('Reading gene.counts')
        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()
        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count

        print("Summary in RARE words DICT")
        for word in ['_RARE_',"_CAPITAL_","_PUNCTUATION_","_NUMBER_","_ONE_","_ContainDOT_","_LongWORDs_","_HeadOfSentence_","_ONENUM_"]:
            for tag in ["O","I-GENE"]:
                print("word: {},tag: {} > {} ".format (word,tag,self.emission_counts[(word,tag)]))
        print("finished reading emission.counts from gene.counts")

    def tagger(self, counts_file, dev_file, output):

        '''
            read counts_file to determine the tag of words, read the dec file,
            then creat output file in '%s %s\n' format
        '''
        #compute_emission(self)

        self.read_counts(counts_file)
        # compute emission parameters
        self.compute_emission()
        print("emission freq of number ,O ")
        self.emission_freq[('_NUMBER_', "O")]
        print("emission freq of number ,I-GENE ")
        self.emission_freq[('_NUMBER_', "I-GENE")]

        word_corpus = set([x[0] for x in self.emission_freq.keys()])
        print("IS new TAG in new list?")
        for word in ['_RARE_',"_CAPITAL_","_PUNCTUATION_","_NUMBER_","_ONE_","_ContainDOT_","_LongWORDs_","_HeadOfSentence_","_ONENUM_"]:
            for tag in ["O","I-GENE"]:
                print("word: {},tag: {} > emmission freq:{} ".format (word,tag,self.emission_freq[(word,tag)]))
        allstate_list =list(self.all_states)
        print('allstate list:',allstate_list)
        #allstate_list =[0, '_RARE_','I-GENE']
        for word in word_corpus:
            counts = []
            for tag in allstate_list:
                counts.append(self.emission_freq[(word,tag)])
                #if word in ['antithrombin','receptors','premature','followed','hepatitis']:
                #print('counts of {} is : {}'.format(word,self.emission_counts[(word,tag)]))
            self.tagger_dict[word] = allstate_list[np.argmax(counts)]
            
        l = dev_file.readline()
        while l:
            word = l[:-1]
            tag =self.tagger_dict[word]
            if word not in list(word_corpus):
                word_tmp = rare_classifier(word)
                #word_tmp = "_RARE_"
                tag = self.tagger_dict[word_tmp]
    
            if l is '\n':
                output.write("\n")
            else:
                output.write("%s %s\n" % (word,tag))
            l = dev_file.readline()
        print('finished writing into evaluation file')

    def compute_emission(self):
        '''
        return a dict which stores emission parameters
        '''

        tag_counts = defaultdict(int)
        for word, ne_tag in self.emission_counts.keys():
            tag_counts[ne_tag] += self.emission_counts[(word, ne_tag)]
        for word, ne_tag in self.emission_counts.keys():
            self.emission_freq[(word, ne_tag)] = \
                self.emission_counts[(word, ne_tag)] / tag_counts[ne_tag]
        
        print('TAG COUNTS: ')
        print(tag_counts)


        return self.emission_freq
    def compute_ngram_freq(self):
        #alltags = list(self.all_states)
        all_state = list(set([key[1] for key in list(self.emission_counts.keys())]))
        print("all_state :",all_state)
        alltags = ['STOP','*']+ all_state
        for y0 in alltags:
            for y1 in alltags:
                for y2 in alltags:
                    if (y0,y1) not in list(self.ngram_counts[1].keys()): pass
                    else:
                        self.ngram_freq[(y2,y0,y1)]= \
                        (self.ngram_counts[2][(y0, y1, y2)] / self.ngram_counts[1][(y0,y1)])
        return self.ngram_freq
    def find_rare(self,n = 5):
        rare_dict =defaultdict(int)
        self.rare_list = []
        for key, value  in self.emission_counts.items():
            rare_dict[key[0]] += value
        for word, count  in rare_dict.items():
            if count < n:
                self.rare_list.append(word)
        print('found {} rare words'.format(len(self.rare_list)))
        return self.rare_list

def usage():
    print ("""
    python count_freqs.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce counts.
        >>> python count_freqs.py gene.train > gene.counts

    """)

if __name__ == "__main__":

    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read input file %s.\n" % arg)
        sys.exit(1)
    
    # Initialize a trigram counter
    counter = Hmm(3)
    # Collect counts

    counter.train(input)
    input.close()
    rare_words = counter.find_rare(n=5)
    #retrain the model

    ######### RETRAIN MODEL WITH RARE WORDS ##########
    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)

    counter1 = Hmm()
    print('Start Retraining with rare words')
    counter1.train(input,rerun = 'True', rare_list = rare_words)
    print('Finished Retraining')
    input.close()

    ########## Write to  gene.counts ##W##########

    print('Start writing to gene.counts')
    count_file_name='gene_AllClass.counts'
    output_file_name = 'gene_dev_AllClass.p2.out'
    input_counts = open( count_file_name,'w')

    counter1.write_counts(input_counts)
    input_counts.close()

    ########### Read gene.counts and write prediction into 'gene_dev.p1.out' #########

    print('Reading gene.counts')
    gene_count = open(count_file_name,"r")
    #counter1.read_counts(gene_count)
    #print(counter1.emission_counts)
    dev_file = open('gene.dev',"r+")

    output_file = open( output_file_name ,"w")
    counter1.train_viterbi( input_counts, dev_file, output_file_name)
    output_file.close()
    gene_count.close()
    dev_file.close()


    #print('starting finding rare words ......')
    
    #print('finished finding rare words.....')
#print(rare_words)
    #input.close()


    #Reading from gene_counts
    
   
    #rare_words = counter.find_rare(n=5)
    
    #for key, count in counter.ngram_freq.items():
    #        print(key,count)
    #for key, freq in counter.emission_freq.items():
    #    print(key,freq)

    ######### Training by Viterbi Algorithm ############
'''
    counter1 = Hmm(3)
    input_counts = open('gene.counts','r')
    dev_file = open('gene.dev',"r+")
    output_file2 = open('gene_dev.p2.out',"w")
    print('dev_file read')
    print('start training viterbi')

    counter1.train_viterbi( input_counts, dev_file, output_file2)
    print('finished training in viterbi')
    dev_file.close()
    output_file2.close()
    input_counts.close()
    print("gene_dev.p2.out file created")
'''


'''

############  Re-train with rare word list  ###############
    print('reopening input file...')
    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)

    counter1 = Hmm(3)
    print('retrain model with _RARE_')

    counter1.train(input,rerun = 'True', rare_list = rare_words)
    print('finish retraining model with _RARE_')
    gene_counts = open('gene.counts',"w")
    counter1.write_counts(gene_counts)
    input.close()
    gene_counts.close()




'''






#a=counter.show_emission()

#print(a)
#print(max(zip(a.values(), a.keys())))
#print(a[('USH1', 'I-GENE')])
