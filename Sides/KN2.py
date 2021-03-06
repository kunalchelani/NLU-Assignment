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

Pkn = dict()
uni = dict()
bi = dict()
tri = dict()

def kneser_ney_setup(train_tokens):

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
		#print "\n"

		f = bigram[0]
		s = bigram[1]
		
		if s not in bi:
			bi[s] = tuple([dict(), dict()])
			bi[s][1]["nov_cnt"] = 1
		else:
			bi[s][1]["nov_cnt"] += 1

		if f not in bi:
			bi[f] = tuple([dict(), dict()])
			bi[f][1]["nov_cnt"] = 0
			bi[f][0][s] = 1
		else:
			if s not in bi[f][0]:
				bi[f][0][s] = 1
			else:
				bi[f][0][s] += 1
	
	#Building Trigram Structure

	for i in range(0, len(train_tokens) - 2):
		
		trigram = train_tokens[i:i+3]
		#print trigram
		#print "\n"

		f = trigram[0]
		s = trigram[1]
		t = trigram[2]
		
		if f not in tri:
			tri[f] = dict()
			tri[f][s] = dict()
			tri[f][s][t] = 1
		else:
			if s not in tri[f]:
				tri[f][s] = dict()
				tri[f][s][t] = 1
			else:
				if t not in tri[f][s]:
					tri[f][s][t] = 1
				else:
					tri[f][s][t] += 1
		
	return


def get_ngram_cnt(ngram):

	tot = 0

	if len(ngram) == 1:

		return uni[ngram[0]]

	elif len(ngram) == 2:

		f = ngram[0]
		s = ngram[1]
		if f in tri:
			if s in tri[f]:
				for t in tri[f][s]:
					tot += tri[f][s][t]

		return tot

	elif len(ngram) == 3:

		f = ngram[0]
		s = ngram[1]
		t = ngram[2]
		if f in tri:
			if s in tri[f]:
				if t in tri[f][s]:
					tot += tri[f][s][t]
		return tot

	else:
		print "Never go full retard  Bro !"


def calc_kn_prob(d, ngram, vocab_size, diff_bigrams):

	#print ngram

	if len(ngram) == 1:
		
		w = ngram[0]  
		pkn = (1.0*bi[w][1]["nov_cnt"])/(diff_bigrams)
		return pkn

	elif len(ngram) == 2:

		pkn_uni = calc_kn_prob(d, [ngram[1]], vocab_size, diff_bigrams)
		
		bigram_cnt = get_ngram_cnt(ngram)
		unigram_cnt = get_ngram_cnt([ngram[0]])

		if ngram[0] in bi:
			lbd = d*len(bi[ngram[0]])

		pkn = ((1.0*max((bigram_cnt - d), 0)) + 1.0*lbd*pkn_uni)/(1.0 * unigram_cnt)
		
		return pkn
	
	elif len(ngram) == 3:

		bigram = ngram[0:2]
		unigram = [ngram[0:1]]
		unigram_cnt = get_ngram_cnt([ngram[0]])
		pkn_bi = calc_kn_prob(d, bigram, vocab_size, diff_bigrams)

		trigram_cnt = get_ngram_cnt(ngram)
		bigram_cnt = pkn_bi*unigram_cnt

		f = ngram[0]
		s = ngram[1]

		lbd = 0
		if f in tri:
			if s in tri[f]:
				lbd = d*len(tri[f][s]) 

		pkn = ((1.0*max((trigram_cnt - d), 0)) + 1.0*lbd*pkn_bi)/(1.0*bigram_cnt)

		return pkn

	'''
	else :
		print "NO"
		return
	'''
	

def calc_kn_perplexity(d, tokens, n):

	vocab_size = len(uni)
	diff_bigrams = 0
	for item in bi:
		diff_bigrams += len(bi[item][0])

	total_words = sum(uni.values())

	tot_log_prob = 0.0

	if (n==2):
		for i in range(0, len(tokens) - 1):
			bigram = tokens[i: i+1]
			prob = calc_kn_prob(d, bigram, vocab_size, diff_bigrams)
			tot_log_prob += math.log(prob)

	if (n==3) :
		for i in range(0, len(tokens) - 2):
			trigram = tokens[i: i+2]
			prob = calc_kn_prob(d, trigram, vocab_size, diff_bigrams)
			tot_log_prob += math.log(prob)
	

	power = (-1.0*tot_log_prob)/len(tokens)
	print "Perplexity : {}".format(math.exp(power))
