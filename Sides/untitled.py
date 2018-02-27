import KN
import ngram
import prep_data
import generate_random as gr

gutenberg_path = '/home/kunal/Downloads/IISc-Acads/sem2/NLU/Assignment-1/Data/gutenberg/'

trigram_object = ngram.N_gram(3, gutenberg_path, thresh = 5)
trigram_object.normalize_train_data()
#trigram_object.normalize_test_data()

KN.kneser_ney_setup(trigram_object.train_tokens)
v = list(KN.uni.keys())
gr.gen_random_sentence(v)