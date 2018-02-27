## Author Kunal Chelani (Kunalchelani@iisc.ac.in)

import numpy as np
import nltk
import random
import math
import sys
from nltk.tokenize import word_tokenize
import ngram
import prep_data
import KN2

class Katz_backoff():

	def __init__(self, train_tokens):
		
		self.katz_setup(train_tokens)

	def katz_setup(self, train_tokens):

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

			if f not in bi:

				bi[f] = tuple([dict(), dict()])
				bi[f][1]["unique_words_succeeding"] = 1
				bi[f][1]["unique_succeeding_sum"] = uni[s]
				bi[f][0][s] = 1

			else:

				if s not in bi[f][0]:

					bi[f][0][s] = 1
					bi[f][1]["unique_words_succeeding"] += 1
					bi[f][1]["unique_succeeding_sum"] +=  uni[s]
				
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
		
		## Assigning to self

		self.uni = uni
		self.bi = bi
		self.tri = tri

		return

	def backoff(self, ngram, delta2, delta3):

		uni = self.uni
		bi  = self.bi
		tri = self.tri

		if len(ngram) == 1:

			print "Why sending all this here"
			
			return 1;
		
		if len(ngram) == 2:

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


		if len(ngram) == 3:

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
							sum_bigram_bo += self.backoff(bigram, delta2, delta3)

						# print "sum_bigram_bo: {}".format(sum_bigram_bo)
 
						prob = ( 1.0 * alpha_fs * self.backoff( ngram[1:3], delta2, delta3 ) ) / (1 - sum_bigram_bo )

						return prob

				else:

					alpha_fs = (1)
					prob = ( 1.0 * alpha_fs * self.backoff( ngram[1:3], delta2, delta3 ) )

					return prob

			else :

				print "Ye to gajab hua ! : {}".format(ngram)
				return 0.0

	def perplexity(self, tokens, delta2, delta3):

		tot_log_prob = 0
		for i in range(0, len(tokens) - 2):
			trigram = tokens[i: i+3]
			prob = self.backoff(trigram,  delta2, delta3)
			#print prob
			tot_log_prob += math.log(prob)

		power = (-1.0*tot_log_prob)/(len(tokens) -2)
		print "Perplexity : {}".format(math.exp(power))

if __name__ == '__main__':

	gutenberg_path = '/home/kunal/Downloads/IISc-Acads/sem2/NLU/Assignment-1/Data/gutenberg/'

	trigram_object = ngram.N_gram(3, gutenberg_path, thresh = 5)
	trigram_object.normalize_train_data()
	trigram_object.normalize_test_data()

	prob_tot = 0
	
	katz_object = Katz_backoff(trigram_object.train_tokens)
	'''
	print katz_object.uni
	print katz_object.bi
	print katz_object.tri

	for word in katz_object.uni.keys():
		trigram = ["sell", "the"] + [word]
		prob = katz_object.backoff(trigram, 0.5, 0.5)
		prob_tot += prob
		print "{} : {}".format(word, prob)
	print prob_tot
	'''
	print "\nTest perplexity"
	katz_object.perplexity(trigram_object.test_tokens, 0.77, 0.77)
	print "\nTrain perplexity"
	katz_object.perplexity(trigram_object.train_tokens, 0.77, 0.77)
	
	



	