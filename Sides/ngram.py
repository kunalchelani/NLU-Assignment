## Author Kunal Chelani (Kunalchelani@iisc.ac.in)

import numpy as np
import os
import io
import re
import nltk
import random
import math
import sys
from nltk.tokenize import word_tokenize
import prep_data

class N_gram():
    
    def __init__(self, n, data_directory, thresh):
        
        if (n==2):
            print "Initializing bigram object"
        elif (n == 3):
            print "Initializing trigram Object"

        (train_data, dev_data, test_data) =  prep_data.prep_gutenberg_data(n, data_directory, 0.7, 0.15)

        train_data = ''.join(train_data)
        train_tokens = word_tokenize(train_data)

        test_data = ''.join(test_data)
        test_tokens = word_tokenize(test_data)

        self.train_tokens = train_tokens
        self.test_tokens= test_tokens
        self.test_size = len(test_tokens)
        self.n = n
        self.thresh = thresh
        return
    
    def normalize_train_data(self):

        print "Normalizing train data"
        train_tokens = self.train_tokens
        freq = prep_data.get_freq_count(train_tokens)
        print "Number of tokens in train = {}".format( len(freq) )

        for index, token in enumerate(train_tokens):
            if freq[token] < self.thresh:
               train_tokens[index] = "UNK" 

        #print "Unknowns in train data :{}".format(count)

        #self.freq = freq
        self.freq = prep_data.get_freq_count(train_tokens)
        self.train_tokens = train_tokens

        fo = open("Train_tokens_normalized.txt", 'w')
        for token in train_tokens:
            print >> fo, "{} ".format(token)
        fo.close()

        #Random Printing
        firstkpairs = {k: freq[k] for k in freq.keys()[:10]}
        print firstkpairs


    def update_ngram_prob(self):
        
        if (self.n==2):
            print "Updating Probabilities for bigrams"
        elif n==3:
            print "Updating Probabilities for trigrams"

        train_tokens = self.train_tokens
        n =  self.n

        cnt = dict()
        prob = dict()
        for i in range(0, len(train_tokens) - n + 1):
            ng = ' '.join(train_tokens[i:i+n])
            if ng not in cnt:
                cnt[ng] = 0
            cnt[ng] += 1
        
        vocab_size = sum(self.freq.values())
        f_probs = open("Bigram Probabilities.txt", 'w')
        for element in cnt:
            prob[element] = (1.0*cnt[element] + 0.00001)/(self.freq[element.split(' ')[0]] + 0.00001*vocab_size)
            f_probs.write("{}\t : {}\t : {}\t:{}\n".format(element, cnt[element], self.freq[element.split(' ')[0]], prob[element]))
        #print prob
        self.prob = prob
        self.vocab_size =  vocab_size	
    
    

    def normalize_test_data(self):

        print "Normalizing test data"

    	test_tokens = self.test_tokens 
        freq = self.freq

        for index, token in enumerate(test_tokens):
            if token not in freq:
                test_tokens[index] = "UNK"
        '''
        fo = open("Test_tokens_normalized.txt", 'w')
        for token in test_tokens:
            print >> fo, "{} ".format(token)
        fo.close()        
        '''
        self.test_tokens = test_tokens

    def calc_data_perplexity(self, tokens, freq, vocab_size, prob, n):
        
        new_prob = dict()
        total_log_prob = 0
        count_there = 0

        for i in range(0, len(tokens) - n + 1):
            ng = ' '.join(tokens[i:i+n])
            if ng not in prob:
                if ng not in new_prob:
                    new_prob[ng] = math.log(0.00001/(freq[ng.split(' ')[0]] + 0.00001*vocab_size))
                pr = new_prob[ng]
            else:
                count_there += 1
                pr = math.log(prob[ng])

            total_log_prob += pr
            #print pr
        print "Tokens there = {}".format(count_there)
        print "Not There = {}".format(len(tokens) - count_there)
        print "No. of tokens : {}".format(len(tokens))        
        print "Total log probability :{}".format(total_log_prob)
        power = (-1.0*total_log_prob)/len(tokens)
        print "power : {}".format(power)

        print math.exp(power)

        #pp = ((1.0/math.exp(total_log_prob)))**power
        pp =1
        print pp
        return pp
    
    def calc_all_perplexity(self):

        print "Calculating All perplexities"

        prob = self.prob
        vocab_size = self.vocab_size
        test_tokens = self.test_tokens
        train_tokens = self.train_tokens
        n  = self.n
        freq =  self.freq

        print "\nTest perplexity data\n"
        self.test_perplexity = self.calc_data_perplexity(test_tokens, freq, vocab_size, prob, n)
        
        print "\nTrain perplexity data\n"
        self.train_perplexity = self.calc_data_perplexity(train_tokens, freq, vocab_size, prob, n)

        return