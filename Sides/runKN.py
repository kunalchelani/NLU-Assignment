import KN2
import ngram
import prep_data

gutenberg_path = '/home/kunal/Downloads/IISc-Acads/sem2/NLU/Assignment-1/Data/gutenberg/'
'''
trigram_object = ngram.N_gram(3, gutenberg_path, thresh = 5)
trigram_object.normalize_train_data()
trigram_object.normalize_test_data()
'''
bigram_object = ngram.N_gram(2, gutenberg_path, thresh = 5)
bigram_object.normalize_train_data()
bigram_object.normalize_test_data()

KN2.kneser_ney_setup(bigram_object.train_tokens)
'''
for i in range (1, 10):
	d = (1.0*i)/10
	print "d = {}".format(d)
	print "\nTest perplexity"
	KN.calc_kn_perplexity(d, trigram_object.test_tokens)
	print "\nTrain perplexity"
	KN.calc_kn_perplexity(d, trigram_object.train_tokens)
	
'''
d = 0.75
print "\nTest perplexity"
KN2.calc_kn_perplexity(d, bigram_object.test_tokens, 2)
print "\nTrain perplexity"
KN2.calc_kn_perplexity(d, bigram_object.train_tokens, 2)