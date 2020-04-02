from count_freqs import *

if __name__ == "__main__":

    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
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
    
    counter1.train(input,rerun = 'True', rare_list = rare_words)
    print('Finished Retraining')
    input.close()

    ########## Write to  gene.counts ##W##########

    
    count_file_name='gene_dev_TwoClass.counts'
    output_file_name = 'gene_dev_TwoClass.p1.out'
    input_counts = open( count_file_name,'w')
    counter1.write_counts(input_counts)
    input_counts.close()

    ########### Read gene.counts and write prediction into 'gene_dev.p1.out' #########

    
    gene_count = open(count_file_name,"r")
    #counter1.read_counts(gene_count)
    #print(counter1.emission_counts)
    dev_file = open('gene.dev',"r+")

    output_file = open( output_file_name ,"w")
    counter1.tagger(gene_count, dev_file, output_file)
    output_file.close()
    gene_count.close()
    dev_file.close()