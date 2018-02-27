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
import ngram
import prep_data
import KN2

class interpolate:

	def __init__(self, l1, l2, l3):
		
		self.l1 = l2
		self.l2 = l2
		self.l3 = l3

		bigram_object = ngram.N_gram(2, gutenberg_path, thresh = 5)
		bigram_object.normalize_train_data()
		bigram_object.normalize_test_data()

		KN2.kneser_ney_setup(bigram_object.train_tokens)
		
		self.tri = KN2.tri
		self.bi = KN2.bi
		self.tri = KN2.uni
		self.vocab_size = KN2.vocab_size
		self.total_words = KN2.total_words
		return

	def calc_probability(trigram):

		bigram  = trigram[0:2]
 		unigram = [trigram[0]]
		tri_cnt = KN2.get_ngram_cnt(trigram)
		bi_cnt = KN2.get_ngram_cnt(bigram)
		uni_cnt = KN2.get_ngram_cnt(unigram)

		if tri_cnt == 0:
			if bi_cnt ==0:
				return self.l3 * (1.0* uni_cnt)/self.total_words
			else:
				return self.l2 * (1.0 * bi_cnt)/(self.vocab_size**2) + self.l3 * (1.0* uni_cnt)/self.total_words
		else :
			return self.l1 * (1.0 * tri_cnt)/(self.vocab_size**3) self.l2 * (1.0 * bi_cnt)/(self.vocab_size**2) + self.l3 * (1.0* uni_cnt)/self.total_words

	def calc_perplexity(data):

		tot_log_prob = 0.0

		for i in range(0, len(data) - 2):
			trigram =  data[i:i+3]
			calc_probability(trigram)
			tot_log_prob += math.log(prob)

		power = (-1.0*tot_log_prob)/len(tokens)
		print "Perplexity : {}".format(math.exp(power))