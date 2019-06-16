import matplotlib
matplotlib.use('Agg')
from model.estimator import Estimator
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import tensorflow as tf
from optparse import OptionParser
from PIL import Image
from models.hmm import HiddenMarkov, load_raw_dataset
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable debug logs Tensorflow.
tf.logging.set_verbosity(tf.logging.ERROR)

# parser = OptionParser()
# parser.add_option("-j", "--json", dest="json_file")
# parser.add_option("-s", "--small", dest="small", action="store_true")
# (options, args) = parser.parse_args()
# options = vars(options)

if __name__ == '__main__':
  estimator = Estimator('data')
  estimator.train()
  estimator.test()

# if sys.argv[1] == 'hmm':
#   start_time = time.time()
#   timesteps = int(sys.argv[2])
#   naive_bayes = timesteps == 0
#   if naive_bayes:
#     timesteps = 1
#   
#   print('Fitting...')
#   # X, Y, _ = load_raw_dataset('data/conll2003_person/train')
#   X, Y, _ = load_raw_dataset('data/ner_on_html/train')
#   hmm = HiddenMarkov(
#     timesteps, 
#     naive_bayes=naive_bayes,
#     use_gazetteer=True,
#     use_features=True,
#     self_train=True
#   )
#   hmm.fit(X, Y)
# 
#   for name in ['train', 'valid', 'test']:
#     print('Predicting ' + name)
#     # x, t, w = load_raw_dataset('data/conll2003_person/' + name)
#     x, t, w = load_raw_dataset('data/ner_on_html/' + name)
#     p = hmm.predict(x)
# 
#     t = [[['O', 'B-PER', 'I-PER'][t__] for t__ in t_] for t_ in t]
#     p = [[['O', 'B-PER', 'I-PER'][p__] for p__ in p_] for p_ in p]
# 
#     with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
#       for words, preds, tags in zip(w, p, t):
#         f.write(b'\n')
#         for word, pred, tag in zip(words, preds, tags):
#           f.write(' '.join([word, tag, pred]).encode() + b'\n')
# 
#   print('Elapsed time: %.4f' % (time.time() - start_time))
