## Author Kunal Chelani (Kunalchelani@iisc.ac.in)

import numpy as np
import os
import io
import re
import nltk
import random
import math
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import sys

def get_freq_count(data):

    freq = dict()
    for token in data:
        if token not in freq:
            freq[token] = 0
        freq[token] += 1
    return freq
    
def prep_gutenberg_data(n, directory, train_split, dev_split):

    print "Preparing data"

    sentences = []
    sentences_nltk = []
    for filename in os.listdir(directory):
        with io.open(directory + filename, 'r', encoding = 'latin-1') as f:
            file_text = f.read()
            sentences_nltk_orig = sent_tokenize(file_text)
            #file_text = re.split('  |\n\n', file_text)
            '''
            for sentence in file_text:
                sentence = sentence.replace('\n', ' ')
                sentence = sentence.replace('_', '')
                if len(sentence) > 0:
                    if sentence[-1] == '.':
                        sentence = sentence[:-1]
                if (n==2):        
                    sentence = "~ {} ~~ ".format(sentence)
                elif (n==3):
                    sentence = "~ ~ {} ~~ ~~ ".format(sentence)
                sentences.append(sentence)
'''
            for sentence in sentences_nltk_orig:
                sentence = sentence.replace('\n', ' ')
                sentence = sentence.replace('_', '')
                #if len(sentence) > 0:
                    #if sentence[-1] == '.':
                        #sentence = sentence[:-1]
                if (n==2):        
                    sentence = "~ {} ~~ ".format(sentence)
                elif (n==3):
                    sentence = "~ ~ {} ~~ ~~ ".format(sentence)
                sentences.append(sentence)
                
    random.shuffle(sentences)
    spl1 = int(math.floor(train_split*len(sentences)))
    spl2 = int(math.floor((train_split + dev_split)*len(sentences)))
    #print spl1
    #print spl2
                
    train_data =  [w.lower() for w in sentences[0:spl1]]
    dev_data =  [w.lower() for w in sentences[spl1:spl2]]
    test_data = [w.lower() for w in sentences[spl2:]]           
    
    return (train_data, dev_data, test_data)

reload(sys)
sys.setdefaultencoding('latin-1')