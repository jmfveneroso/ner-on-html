import sys
import math
import numpy as np
from crf import Crf, InputLayer
from bi_rnn import SingleRNN, BiRNN

labels       = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
observations = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])
# labels       = np.array([0, 1, 0, 0])
# observations = np.array([2, 1, 2, 3])
t = len(observations)
n_x = 3
n_y = 2
n_a = 20

def one_hot(target, num_classes):
  v = np.zeros(num_classes)
  v[target] = 1
  return v

def create_dataset():
  X = np.zeros((3, t))
  Y = np.zeros((2, t))
  for i in range(t):
    X[:,i] = one_hot(observations[i] - 1, 3)
    Y[:,i] = one_hot(labels[i], 2)
  return X, Y

X, Y = create_dataset()

rnn = BiRNN(n_x, n_y, n_a, t)
rnn.set_input(X)
crf = Crf(n_y, t, rnn)
crf.fit(Y, 0.01, 2000)
print labels.tolist()
print crf.viterbi(X)

# y_pred = rnn.fit(X, Y)
# print np.argmax(y_pred, axis=0)
# print labels
