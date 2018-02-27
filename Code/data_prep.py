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

reload(sys)
sys.setdefaultencoding('latin-1')

def process_sentences(sentences, n):

	new_sentences = []
	for sent in sentences:

		new_sent = []
		
		if n==2:

			new_sent.append("~")

			for word in sent:
				new_sent.append((str(word)).lower())

			new_sent.append("~")

		elif  n==3:

			new_sent.append("~")
			new_sent.append("~")
			
			for word in sent:
				new_sent.append((str(word)).lower())

			new_sent.append("~")
			new_sent.append("~")

		else :
			print "Not handled right now"
			exit(0)

		new_sentences.append(new_sent)

	return new_sentences

def get_initial_freq(data):

	freq = dict()
	for token in data:
		if token not in freq:
			freq[token] = 0

		freq[token] += 1

	return freq


def normalize_train_data(train_tokens, freq, thresh):

	#print "Normalizing train data"
	#print "Number of tokens in train = {}".format( len(freq) )

	for index, token in enumerate(train_tokens):
		if freq[token] < thresh:
			train_tokens[index] = "UNK"

	fo = open("Train_tokens_normalized.txt", 'w')
	for token in train_tokens:
	    print >> fo, "{} ".format(token)
	fo.close()

	return train_tokens


def normalize_test_data(test_tokens, freq):
	
	#print "Normalizing test data"

	for index, token in enumerate(test_tokens):
		if token not in freq:
			"replacing word : {}".format(token)
			test_tokens[index] = "UNK"

	return test_tokens	


def split_brown_data(sentences, train_split, dev_split):

	random.shuffle(sentences)
	spl1 = int(math.floor(train_split*len(sentences)))
	spl2 = int(math.floor((train_split + dev_split)*len(sentences)))
	
	train_data = sentences[0:spl1]
	dev_data = sentences[spl1:spl2]
	test_data = sentences[spl2:]

	return (train_data, dev_data, test_data)


def split_gutenberg_data(sentences, train_split, dev_split):

	random.shuffle(sentences)
	spl1 = int(math.floor(train_split*len(sentences)))
	spl2 = int(math.floor((train_split + dev_split)*len(sentences)))

	train_data =  [w.lower() for w in sentences[0:spl1]]
	dev_data =  [w.lower() for w in sentences[spl1:spl2]]
	test_data = [w.lower() for w in sentences[spl2:]]
	
	return (train_data, dev_data, test_data)           
    


def prep_brown_data(n, train_split, dev_split):

	print "Preparing Brown data"
	
	cat_orig = brown.categories()
	cat_new = []
	for cat in cat_orig:
		cat_new.append(str(cat))

	sentences = brown.sents(categories = cat_new)
	
	new_sentences =  process_sentences(sentences, n)

	(train_data, dev_data, test_data) = split_brown_data(new_sentences, train_split, dev_split)

	train_tokens = []
	dev_tokens = []
	test_tokens = []

	for sent in train_data: 
		train_tokens += sent

	for sent in dev_data: 
		dev_tokens += sent

	for sent in test_data: 
		test_tokens += sent

	return (train_tokens, dev_tokens, test_tokens)


def prep_gutenberg_data(n, directory, train_split, dev_split):

    print "Preparing Gutenberg data"

    sentences = []

    for filename in os.listdir(directory):

        with io.open(directory + filename, 'r', encoding = 'latin-1') as f:
            file_text = f.read()

            sentences_nltk_orig = sent_tokenize(file_text)

            for sentence in sentences_nltk_orig:

                sentence = sentence.replace('\n', ' ')
                sentence = sentence.replace('_', '')

                if (n==2):        
                    sentence = "~ {} ~~ ".format(sentence)
                elif (n==3):
                    sentence = "~ ~ {} ~~ ~~ ".format(sentence)
                sentences.append(sentence)


    (train_data, dev_data, test_data) = split_gutenberg_data(sentences, train_split, dev_split)
                
    train_data = ''.join(train_data)
    train_tokens = word_tokenize(train_data)

    dev_data = ''.join(dev_data)
    dev_tokens = word_tokenize(dev_data)

    test_data = ''.join(test_data)
    test_tokens = word_tokenize(test_data)

    return (train_tokens, dev_tokens, test_tokens)

# For testing purposes
'''
if __name__ == '__main__' :

	gutenberg_path = '/home/kunal/Downloads/IISc-Acads/sem2/NLU/Assignment-1/Data/gutenberg/'
	thresh = 5

	(brown_train_tokens, brown_dev_tokens, brown_test_tokens) = prep_brown_data(3, 0.80, 0.10)
	(gut_train_tokens, gut_dev_tokens, gut_test_tokens) =  prep_gutenberg_data(3, gutenberg_path, 0.80, 0.10)

	freq_brown = get_initial_freq(brown_train_tokens)
	freq_gut = get_initial_freq(gut_train_tokens)

	normalized_brown_train_tokens = normalize_train_data(brown_train_tokens, freq_brown, thresh)
	normalized_gut_test_tokens = normalize_test_data(gut_test_tokens, freq_gut)

	print normalized_brown_train_tokens[0:50]
	print normalized_gut_test_tokens[0:500]
'''