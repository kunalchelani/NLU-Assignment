import os
import io
import re
import nltk
import random
import math
import sys
from nltk.tokenize import word_tokenize
import ngram
import prep_data
import KN2
import numpy as np


d = 0.75
prob = dict()

def gen_random_sentence(v):

	vocab_size = len(KN2.uni)
	diff_bigrams = 0
	for item in KN2.bi:
		diff_bigrams += len(KN2.bi[item][0])
	
	
	#words_gen.append("~")
	
	next_word = ""
	for k in range(0,10):
		i = 0
		words_gen = []
		words_gen.append("~")
		words_gen.append("~")
		while words_gen[-1] != "~~":

			#print words_gen
			for token in v:
				#print token
				trigram = words_gen[i: i+2] + [token]
				prob[token] = KN2.calc_kn_prob(d, trigram, vocab_size, diff_bigrams)

			prob_sum =  sum(prob.values())
			#print prob_sum

			for token in prob:
				prob[token] = float(prob[token])/prob_sum

			tok = np.random.choice(prob.keys(), 1, p=prob.values())
			if tok not in ["UNK", "~"]:
				words_gen.append(tok[0])

				#print words_gen

				i += 1

		print ' '.join(words_gen[0:-1])
		print "\n"

		
gutenberg_path = '/home/kunal/Downloads/IISc-Acads/sem2/NLU/Assignment-1/Data/gutenberg/'

trigram_object = ngram.N_gram(3, gutenberg_path, thresh = 5)
trigram_object.normalize_train_data()
#bigram_object.normalize_test_data()

KN2.kneser_ney_setup(trigram_object.train_tokens)
gen_random_sentence(KN2.uni.keys())					

	