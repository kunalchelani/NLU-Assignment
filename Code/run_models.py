## Author Kunal Chelani ( kunalchelani@iisc.ac.in )

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import brown
import random
import math
import sys
import os
import io
import data_prep as dp
import lang_models as lm
import matplotlib.pyplot as plt

#r = 1
#sett = [0,0,0,0]

def perform_experiment(train_data, test_data, dev_data, delta2, delta3, d, k):

	counts = lm.setup_counts(train_data)

	## Hperpaprameter tuning for Katz
	
	dev_perplexities =  np.zeros(10)
	ks = np.zeros(10)
	
	#global r
	for i in range(1, 10):
		delta2 = 1.0*i/10
		delta3 = 1.0*i/10
		ks[i] = 1.0*i/10
		dev_perplexities[i] = lm.katz_perplexity(tokens = dev_data, counts = counts, delta2 = delta2, delta3 = delta3)
		print "katz perplexity : {}".format(dev_perplexities[i])

	#global sett	
	#sett[r-1], = plt.plot( np.log(ks[1:9]), dev_perplexities[1:9], label = "Setting : {}".format(r))

	
	## Hyperparameter tuning for Add_k
	#global r

	dev_perplexities =  np.zeros(8)
	ks = np.zeros(8)
	
	for i in range(1,8):
		k = 10**(-1*i)
		pp = lm.add_k_perplexity(tokens = dev_data, counts = counts, k = k)
		ks[i] = k	
		dev_perplexities[i] = (pp)
		print "add_k_perplexity for k {} : {}".format(k, pp)

	#global sett	
	#sett[r-1], = plt.plot( np.log(ks[1:8]), dev_perplexities[1:8], label = "Setting : {}".format(r))

	#r = r+1

	add_k_perplexity = lm.add_k_perplexity(tokens = test_data, counts = counts, k = k)

	## Laplacian

	k = 1

	laplacian_perplexity = lm.add_k_perplexity(tokens = test_data, counts = counts, k = k)
	print "laplacian_perplexity : {}".format(laplacian_perplexity)
	
	## Kneser ney
	#global r
	#global sett
	
	dev_perplexities =  np.zeros(10)
	ks = np.zeros(10)

	bi = counts[1]
	diff_bigrams = 0

	for item in bi:
		diff_bigrams += len(bi[item][0])

	for i in range(1, 10):
		d = (1.0*i)/10
		ks[i] = d
		dev_perplexities[i] = lm.kn_perplexity(tokens = test_data, counts = counts , diff_bigrams = diff_bigrams, d=d)
		print "kn_perplexity : {}".format(dev_perplexities[i])

	#sett[r-1], = plt.plot( np.log(ks[1:8]), dev_perplexities[1:8], label = "Setting : {}".format(r))



if __name__ == '__main__':

	## All Data Preparation part. At the end of this section we have all normalized data sets

	if sys.argv[1] == "pp":

		gutenberg_path = '../Data/gutenberg/'
		thresh = 7

		(brown_train_tokens, brown_dev_tokens, brown_test_tokens) = dp.prep_brown_data(3, 0.70, 0.15)
		(gut_train_tokens, gut_dev_tokens, gut_test_tokens) =  dp.prep_gutenberg_data(3, gutenberg_path, 0.70, 0.15)

	## This is where we run tests on the four settings given as part of the assignment

	

		## Setting 1: Train: D1-Train, Test: D1-Test

		print "\nSetting 1 Results\n"

		test_data = brown_test_tokens
		dev_data = brown_dev_tokens
		train_data = brown_train_tokens

		freq_train = dp.get_initial_freq(train_data)
		n_train_data = dp.normalize_train_data(train_data, freq_train, thresh)
		n_freq_train = dp.get_initial_freq(n_train_data)
		n_dev_data = dp.normalize_test_data(dev_data, n_freq_train)
		n_test_data = dp.normalize_test_data(test_data, n_freq_train)

		delta2 = (1.5 * 5 )/10.0
		delta3 = (1.5 * 5 )/10.0
		k = 0.001
		d = 0.75

		perform_experiment(train_data = n_train_data, test_data = n_test_data, dev_data = n_dev_data, delta2 = delta2, delta3 = delta3,d = d, k = k)

		## Setting 2: Train: D2-Train, Test: D2-Test

		print "\nSetting 2 Results\n"

		test_data = gut_test_tokens
		dev_data = gut_dev_tokens
		train_data = gut_train_tokens

		freq_train = dp.get_initial_freq(train_data)
		n_train_data = dp.normalize_train_data(train_data, freq_train, thresh)
		n_freq_train = dp.get_initial_freq(n_train_data)
		n_dev_data = dp.normalize_test_data(dev_data, n_freq_train)
		n_test_data = dp.normalize_test_data(test_data, n_freq_train)


		delta2 = (1.5 * 5 )/10.0
		delta3 = (1.5 * 5 )/10.0
		k = 0.001
		d = 0.75

		perform_experiment(train_data = n_train_data, test_data = n_test_data, dev_data = n_dev_data, delta2 = delta2, delta3 = delta3,d = d, k = k)



		## Setting 3 : Train: D1-Train + D2-Train, Test: D1-Test

		print "\nSetting 3 Results\n"

		test_data = brown_test_tokens
		dev_data = brown_dev_tokens + gut_dev_tokens
		train_data = brown_train_tokens + gut_train_tokens

		freq_train = dp.get_initial_freq(train_data)
		n_train_data = dp.normalize_train_data(train_data, freq_train, thresh)
		n_freq_train = dp.get_initial_freq(n_train_data)
		n_dev_data = dp.normalize_test_data(dev_data, n_freq_train)
		n_test_data = dp.normalize_test_data(test_data, n_freq_train)

		delta2 = (1.5 * 5 )/10.0
		delta3 = (1.5 * 5 )/10.0
		k = 0.001
		d = 0.75

		perform_experiment(train_data = n_train_data, test_data = n_test_data,dev_data = n_dev_data, delta2 = delta2, delta3 = delta3,d = d, k = k)

		print "\nSetting 4 Results\n"

		test_data = gut_test_tokens
		dev_data = brown_dev_tokens + gut_dev_tokens
		train_data = brown_train_tokens + gut_train_tokens

		freq_train = dp.get_initial_freq(train_data)
		n_train_data = dp.normalize_train_data(train_data, freq_train, thresh)
		n_freq_train = dp.get_initial_freq(n_train_data)
		n_dev_data = dp.normalize_test_data(dev_data, n_freq_train)
		n_test_data = dp.normalize_test_data(test_data, n_freq_train)


		delta2 = (1.5 * 5 )/10.0
		delta3 = (1.5 * 5 )/10.0
		k = 0.001
		d = 0.75

		perform_experiment(train_data = n_train_data, test_data = n_test_data,dev_data = n_dev_data, delta2 = delta2, delta3 = delta3,d = d, k = k)

		plt.legend(handles = [sett[0], sett[1], sett[2], sett[3]])
		plt.xlabel("Perplexity")
		plt.ylabel("delta")
		plt.title("Hyperparameter tuning for Katz with equal deltas")
		plt.savefig('Katz.jpeg')


	elif sys.argv[1] == "gen":

		gutenberg_path = '../Data/gutenberg/'
		(gut_train_tokens, gut_dev_tokens, gut_test_tokens) =  dp.prep_gutenberg_data(3, gutenberg_path, 0.70, 0.15)
		
		print "generating"

		thresh = 5
		train_data = gut_train_tokens
		freq_train = dp.get_initial_freq(train_data)
		n_train_data = dp.normalize_train_data(train_data, freq_train, thresh)
		
		counts = lm.setup_counts(n_train_data)

		prob = dict()
		uni = counts[0]
		bi = counts[1]
		tri = counts[2]

		num_sentences = 1

		for k in range(0,  num_sentences):
			words_gen = []
			words_gen.append("~")
			words_gen.append("~")
			i = 0
			while len(words_gen) < 11:

				# print words_gen
				valid_set = set(uni.keys()) - set(["UNK", "~", "~~"])
				for token in valid_set:
					gram = words_gen[i: i+2] + [token]
					prob[token] = lm.backoff(gram, counts, 0.75, 0.75)

				prob_sum = sum(prob.values())
				# print prob_sum

				for token in prob:
					prob[token] = float(prob[token])/prob_sum

				tok = np.random.choice(prob.keys(), 1, p=prob.values())
				while tok in ["UNK", "~", "~~"]:
					#print "Generating new one"
					tok = np.random.choice(prob.keys(), 1, p=prob.values()) 
				
				words_gen.append(tok[0])
				i = i+1

				#print words_gen

			print "\nSentence :\n"	
			print ' '.join(words_gen[2:-1])
			print "\n"

		"Generating from Kneser-ney"

		words_gen = []
		words_gen.append("~")
		words_gen.append("~")
		i = 0
		while len(words_gen) < 11:

			# print words_gen
			valid_set = set(uni.keys()) - set(["UNK", "~", "~~"])
			for token in valid_set:
				gram = words_gen[i: i+1] + [token]
				prob[token] = lm.kn_prob(gram, counts, 0.75, 0.75)

			prob_sum = sum(prob.values())
			# print prob_sum

			for token in prob:
				prob[token] = float(prob[token])/prob_sum

			tok = np.random.choice(prob.keys(), 1, p=prob.values())
			while tok in ["UNK", "~", "~~"]:
				#print "Generating new one"
				tok = np.random.choice(prob.keys(), 1, p=prob.values()) 
			
			words_gen.append(tok[0])
			i = i+1

		print "\nSentence :\n"	
		print ' '.join(words_gen[2:-1])
		print "\n"

