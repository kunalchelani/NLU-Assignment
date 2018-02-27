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

def setup_counts(train_tokens):

	uni = dict()
	bi = dict()
	tri = dict()

	# Building Unigram Structure

	for i in range(0, len(train_tokens)):
		
		unigram = train_tokens[i] 
		if unigram not in uni:
			uni[unigram] = 0
		uni[unigram] += 1

	# Building Bigram structure

	for i in range(0, len(train_tokens) - 1):
		
		bigram = train_tokens[i:i+2]
		#print bigram
		
		f = bigram[0]
		s = bigram[1]
		
		if s not in bi:

			bi[s] = tuple([dict(), dict()])
			bi[s][1]["unique_words_succeeding"] = 0
			bi[s][1]["unique_succeeding_sum"] = 0
			bi[s][1]["nov_cnt"] = 1

		if f not in bi:

			bi[f] = tuple([dict(), dict()])
			bi[f][1]["nov_cnt"] = 0
			bi[f][1]["unique_words_succeeding"] = 1
			bi[f][1]["unique_succeeding_sum"] = uni[s]
			bi[f][0][s] = 1

		else:

			if s not in bi[f][0]:

				bi[f][0][s] = 1
				bi[f][1]["unique_words_succeeding"] += 1
				bi[f][1]["unique_succeeding_sum"] +=  uni[s]
				bi[s][1]["nov_cnt"] += 1
			
			else:

				bi[f][0][s] += 1

	#Building Trigram Structure

	for i in range(0, len(train_tokens) - 2):

		trigram = train_tokens[i:i+3]
		# print trigram

		f = trigram[0]
		s = trigram[1]
		t = trigram[2]
		
		if f not in tri:
			tri[f] = dict()
			tri[f][s] = tuple([dict(), dict()])
			tri[f][s][1]["unique_words_succeeding"] = 1
			tri[f][s][1]["unique_succeeding_sum"] = uni[t]
			tri[f][s][0][t] = 1
		else:
			if s not in tri[f]:
				tri[f][s] = tuple([dict(), dict()])
				tri[f][s][0][t] = 1
				tri[f][s][1]["unique_words_succeeding"] = 1
				tri[f][s][1]["unique_succeeding_sum"] = uni[t]
			else:
				if t not in tri[f][s][0]:
					tri[f][s][0][t] = 1
					tri[f][s][1]["unique_words_succeeding"] += 1
					tri[f][s][1]["unique_succeeding_sum"] += uni[t]
				else:
					tri[f][s][0][t] += 1

	counts = tuple((uni, bi, tri))
	return counts


def backoff(ngram, counts, delta2, delta3):

	#print ngram

	uni = counts[0]
	bi  = counts[1]
	tri = counts[2]

	if len(ngram) == 1:

		print "Why sending all this here"
		
		return 1;

	elif len(ngram) == 2:

		f = ngram[0]
		s = ngram[1]

		if f in bi:

			if s in bi[f][0]:

				prob = (1.0 * (bi[f][0][s] - delta2))/ uni[f]
				#print "prop_prob : {}, {}, {}".format(ngram, bi[f][0][s], uni[f])
				return prob

			else:

				#print "Bigram not found, backing off"

				alpha_f = ( bi[f][1]["unique_words_succeeding"] * delta2 * 1.0) / uni[f]

				#print "alpha_f : {}".format(alpha_f)

				prob = (alpha_f * uni[s] * 1.0) / ( bi[f][1]["unique_succeeding_sum"] )
				
				#print "bo_prob : {}, {}".format(prob, ngram)
				return prob

		else:

			print "The unigram was not found, this is weird "
			
			return 0


	elif len(ngram) == 3:

		f = ngram[0]	
		s = ngram[1]
		t = ngram[2]
		
		if f in tri:
			
			if s in tri[f]:
				
				if t in tri[f][s][0]:
					
					prob = ( 1.0 * ( tri[f][s][0][t] - delta3 ) ) / ( bi[f][0][s] )

					return prob

				else:

					# print "backing off to bigrams"
					
					alpha_fs = (tri[f][s][1]["unique_words_succeeding"] * delta3) / bi[f][0][s]

					#print "alpha_fs : {}".format(alpha_fs)

					sum_bigram_bo = 0

					for t in tri[f][s][0]:
						bigram = [s] +[t]
						sum_bigram_bo += backoff(bigram, counts, delta2, delta3)

					# print "sum_bigram_bo: {}".format(sum_bigram_bo)

					prob = ( 1.0 * alpha_fs * backoff( ngram[1:3], counts, delta2, delta3 ) ) / (1 - sum_bigram_bo )

					return prob

			else:

				alpha_fs = (1)
				prob = ( 1.0 * alpha_fs * backoff( ngram[1:3], counts, delta2, delta3 ) )

				return prob

		else :

			print "Ye to gajab hua ! : {}".format(ngram)
			return 0.0
	
	else :

		print "Not right now"
		return 0


def laplacian_prob(ngram, counts):

	uni = counts[0]
	bi  = counts[1]
	tri = counts[2]

	vocab_size = len(uni) 

	if len(ngram) == 1:

		f = ngram[0]
		prob = (1.0*uni[f] + 1.0) / (2.0 * vocab_size)   

	if len(ngram) == 2:

		f = ngram[0]
		s = ngram[1]

		if s in bi[f][0]:

			prob = ( 1.0 * ( bi[f][0][s] + 1.0 ) ) / ( 1.0 * (uni[f]) + vocab_size )

		else :

			prob = (1.0) / ( 1.0 * (uni[f]) + vocab_size )
			print prob

	return prob


def add_k_prob(ngram, counts, k):

	uni = counts[0]
	bi  = counts[1]
	tri = counts[2]

	vocab_size = len(uni)

	if len(ngram) == 1:

		f = ngram[0]
		prob = (1.0*uni[f] + 1.0) / ((1+k) * vocab_size) 

	if len(ngram) == 2:

		f = ngram[0]
		s = ngram[1]

		if s in bi[f][0]:

			prob = ( 1.0 * ( bi[f][0][s] + k ) ) / ( 1.0 * (uni[f]) + k * vocab_size )

		else :

			prob = 1.0 * k / ( 1.0 * (uni[f]) + k * vocab_size )

	return prob


def kn_prob(ngram, counts, diff_bigrams, d):

	uni = counts[0]
	bi = counts[1]
	tri = counts[2]

	vocab_size = len(uni)

	if len(ngram) == 1:
		
		w = ngram[0]  
		pkn = ( 1.0 * bi[w][1]["nov_cnt"] )/(diff_bigrams)
		return pkn

	elif len(ngram) == 2:
		
		f = ngram[0]
		s = ngram[1]

		p_cont = kn_prob([ngram[1]], counts, diff_bigrams, d)
		
		if s in bi[f][0]:
			bigram_cnt = bi[f][0][s]
		else:
			bigram_cnt = 0
		
		unigram_cnt = uni[f]

		lbd = (1.0 * d * len(bi[f][0]))

		pkn =  (( 1.0 * max( (bigram_cnt - d), 0) ) + (1.0*lbd*p_cont ) )  / (1.0 * unigram_cnt)
		
		return pkn

	elif len(ngram) == 3:
		
		f = ngram[0]
		s = ngram[1]
		t = ngram[2]

		bigram_bo = ngram[1:3]
		bigram = ngram[0:2]
		
		unigram_cnt = uni[s]

		pkn_bi_bo = kn_prob(bigram_bo, counts, diff_bigrams, d)
		pkn_bi_1 = kn_prob(bigram, counts, diff_bigrams, d)
 		trigram_cnt = 0

		if s in tri[f]:
			if t in tri[f][s][0]:
				trigram_cnt = tri[f][s][0][t]

		bigram_cnt = pkn_bi_1 * uni[f]

		lbd = 0.0001
		if f in tri:
			if s in tri[f]:
				lbd = d*len(tri[f][s][0]) 

		pkn = ( (1.0 * max ( (trigram_cnt - d), 0) ) + (1.0* lbd* pkn_bi_bo) ) / (1.0 * bigram_cnt)

		return pkn


def add_k_perplexity(tokens, counts, k):
	
	tot_log_prob = 0
	
	for i in range(0, len(tokens) - 1):

		bigram = tokens[i: i+2]
		prob = add_k_prob(bigram, counts, k)
		tot_log_prob += math.log(prob)

	power = (-1.0*tot_log_prob)/(len(tokens) - 1)

	return math.exp(power)


def laplacian_perplexity(tokens, counts, k):

	tot_log_prob = 0
	
	for i in range(0, len(tokens) - 1):

		bigram = tokens[i: i+2]
		prob = laplacian_prob(bigram, counts)
		tot_log_prob += math.log(prob)

	power = (-1.0*tot_log_prob)/(len(tokens) - 1)

	return math.exp(power)


def katz_perplexity(tokens, counts, delta2 = 0.5, delta3 = 0.5):

	tot_log_prob = 0
	
	for i in range(0, len(tokens) - 2):

		trigram = tokens[i: i+3]
		prob = backoff(trigram, counts,  delta2, delta3)
		tot_log_prob += math.log(prob)

	power = (-1.0*tot_log_prob)/(len(tokens) -2)

	return math.exp(power)


def kn_perplexity(tokens, counts, diff_bigrams, d):

	tot_log_prob = 0
	
	for i in range(0, len(tokens) - 2):

		bigram = tokens[i: i+3]
		prob = kn_prob(bigram, counts, diff_bigrams, d)	
		tot_log_prob += math.log(prob)

	power = (-1.0*tot_log_prob)/(len(tokens) -2)

	return math.exp(power)