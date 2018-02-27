import nltk
import random
import math
import sys
from nltk.tokenize import word_tokenize
import ngram
import prep_data
import numpy as np
import katz

prob = dict()
inf = 1
def gen_random_sentence(katz_object):

	num_sentences = 5
	if (inf == 0):
		for k in range(0,  num_sentences):
			words_gen = []
			words_gen.append("~")
			words_gen.append("~")
			i = 0
			while len(words_gen) < 12:

				#print words_gen
				for token in katz_object.uni.keys():
					#print token
					trigram = words_gen[i: i+2] + [token]
					prob[token] = katz_object.backoff(trigram, 0.5, 0.5)

				prob_sum = sum(prob.values())
				#print prob_sum

				for token in prob:
					prob[token] = float(prob[token])/prob_sum

				tok = np.random.choice(prob.keys(), 1, p=prob.values())
				while tok in ["UNK", "~"]:
					"Generating new one"
					tok = np.random.choice(prob.keys(), 1, p=prob.values())
				
				words_gen.append(tok[0])
				i = i+1

				#print words_gen

			print ' '.join(words_gen[0:-1])
			print "\n"

	else:

		
		
		for k in range(0,10):
			i = 0
			words_gen = []
			words_gen.append("~")
			words_gen.append("~")
			while words_gen[-1] != "~~":
				for token in katz_object.uni.keys():
					#print token
					trigram = words_gen[i: i+2] + [token]
					prob[token] = katz_object.backoff(trigram, 0.5, 0.5)

				prob_sum = sum(prob.values())
				#print prob_sum

				for token in prob:
					prob[token] = float(prob[token])/prob_sum

				tok = np.random.choice(prob.keys(), 1, p=prob.values())
				while tok in ["UNK", "~"]:
					"Generating new one"
					tok = np.random.choice(prob.keys(), 1, p=prob.values())
				
				words_gen.append(tok[0])
				i = i+1

				#print words_gen

			print ' '.join(words_gen[2:-1])
			print "\n"


if __name__ == '__main__':

	gutenberg_path = '/home/kunal/Downloads/IISc-Acads/sem2/NLU/Assignment-1/Data/gutenberg/'

	trigram_object = ngram.N_gram(3, gutenberg_path, thresh = 5)
	trigram_object.normalize_train_data()


	katz_object = katz.Katz_backoff(trigram_object.train_tokens)

	gen_random_sentence(katz_object)


