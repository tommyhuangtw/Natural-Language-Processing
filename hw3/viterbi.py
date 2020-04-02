from count_freqs import *
from eval_gene_tagger import *
'''
Using gene.train gene.counts prediction file to evaluate the performance

Usage: python viterbi.py gene.counts gene.dev gene_dev.p1.out 
'''

if __name__ == "__main__":

    #if len(sys.argv)!=2: # Expect exactly one argument: the training data file
    #    usage()
    #    sys.exit(2)

    #try:
    #    input_counts = open(sys.argv[1],"r")
    #    dev_file = open(sys.argv[2],"r+")
    #    output_file2 = open(sys.argv[3],"w")
    #except IOError:
    #    sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
    #    sys.exit(1)

    
 


    ########### Read gene.counts and write prediction into 'gene_dev.p1.out' #########

    counter1 = Hmm(3)
    input_counts = open('gene_NoClass.counts','r')
    dev_file = open('gene.dev',"r+")
    output_file2 = open('gene_dev.NoClass.out.p2',"w")
    print('dev_file read')
    print('start training viterbi')


    counter1.train_viterbi( input_counts, dev_file, output_file2)
    print('finished training in viterbi')
    dev_file.close()
    output_file2.close()
    input_counts.close()
    print("gene_dev.p2.out file created")







    ######### Evaluate the result ############


    '''
    if len(sys.argv)!=3:
        usage()
        sys.exit(1)
    gs_iterator = corpus_iterator(open(sys.argv[1]))
    pred_iterator = corpus_iterator(open(sys.argv[4]), with_logprob = False)
    evaluator = Evaluator()
    evaluator.compare(gs_iterator, pred_iterator)
    evaluator.print_scores()

    '''