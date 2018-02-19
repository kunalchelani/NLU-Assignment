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

# Script Starts Here

gutenberg_path = '/home/kunal/Downloads/IISc-Acads/sem2/NLU/Assignment-1/Data/gutenberg/'

bigram = ngram.N_gram(2, gutenberg_path)

bigram.normalize_train_data()
bigram.update_ngram_prob()
bigram.normalize_test_data()
bigram.calc_all_perplexity()

print bigram.train_perplexity
print bigram.test_perplexity