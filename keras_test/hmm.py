import np
import re
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import sys
import numpy as np

X = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])
Y = np.array([1, 1, 2, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2])

labels = np.array(['O', 'B-NAME', 'I-NAME'])

num_labels = 3
time_steps = 3
vocab_size = 3

num_states = num_labels ** time_steps
transition_mat = np.ones((num_states, num_labels))
emission_mat = np.ones((vocab_size, num_labels))

start = np.zeros((num_states, 1))
start[0,:] = 1
end = np.ones((num_states, 1))

def idx_to_states(idx):
  states = []
  multiplier = num_labels ** (time_steps - 1)
  for i in range(time_steps):
    states.append(int(idx) // int(multiplier))
    idx %= multiplier
    multiplier /= num_labels
  return states 

def states_to_idx(states):
  if len(states) < time_steps:
    raise Exception('Wrong states length.')

  acc = 0
  multiplier = 1
  for s in reversed(states):
    acc += int(multiplier) * int(s)
    multiplier *= num_labels
  return acc

def load_gazetteer(name_file, word_file):
  o, b_name, i_name = {'$UNK': 1}, {'$UNK': 1}, {'$UNK': 1}
  with open(name_file) as f:
    for line in f:
      tkns = line.strip().split(' ')
      if len(tkns) == 0:
        continue

      if not tkns[0] in b_name:
        b_name[tkns[0]] = 2 # Laplace smoothing.
      else:
        b_name[tkns[0]] += 1

      for t in tkns[1:]: 
        if not t in i_name:
          i_name[t] = 2
        else:
          i_name[t] += 1

  with open(word_file) as f:
    for line in f:
      tkns = line.strip().split(' ')
      for t in tkns: 
        if not t in o:
          o[t] = 2
        else:
          o[t] += 1

  o_count = sum([o[tkn] for tkn in o])
  for tkn in o:
    o[tkn] = float(o[tkn]) / float(o_count)

  b_name_count = sum([b_name[tkn] for tkn in b_name])
  for tkn in b_name:
    b_name[tkn] = float(b_name[tkn]) / float(b_name_count)

  i_name_count = sum([i_name[tkn] for tkn in i_name])
  for tkn in i_name:
    i_name[tkn] = float(i_name[tkn]) / float(i_name_count)

def fit(X):
  global transition_mat, emission_mat
  states = [0] * time_steps
  for i in range(len(X)):
    idx = states_to_idx(states)
    transition_mat[idx,Y[i]] += 1
    emission_mat[X[i]-1,Y[i]] += 1
    states.pop(0) 
    states.append(Y[i]) 
  transition_mat /= np.expand_dims(np.sum(transition_mat, axis=1), axis=1)
  transition_mat = np.nan_to_num(transition_mat)

  emission_mat = emission_mat.T
  emission_mat /= np.expand_dims(np.sum(emission_mat, axis=1), axis=1)
  emission_mat = np.nan_to_num(emission_mat.T)

def viterbi(X):
  pointers = np.zeros((len(X), num_states), dtype=int)

  state_probs = start 
  for i in range(len(X)):
    p = state_probs * transition_mat * emission_mat[X[i]-1]
    state_probs = np.zeros((num_states, 1))
    for s in range(num_states):
      for l in range(num_labels):
        states = idx_to_states(s)
        states.pop(0)
        states.append(l)
        idx = states_to_idx(states)
         
        if p[s,l] > state_probs[idx,:]:
          pointers[i,idx] = s
          state_probs[idx,0] = p[s,l]

  idx = np.argmax(state_probs)
  labels = [] 
  for i in reversed(range(len(X))):
    states = idx_to_states(idx)
    labels.append(states[-1])
    idx = pointers[i,idx]
  return list(reversed(labels))

load_gazetteer('names.txt', 'words.txt')

# fit(X)
# print(transition_mat)
# print(emission_mat)
# 
# print(list(Y))
# print(viterbi(X))
# 
# from util import Dataset, WordEmbeddings
# 
# we       = WordEmbeddings('glove-50')
# dev      = Dataset(we, 'dataset/dev.txt', max_sentence_len=50, max_token_len=20)
# validate = Dataset(we, 'dataset/validate.txt', max_sentence_len=50, max_token_len=20)
# test     = Dataset(we, 'dataset/test.txt', max_sentence_len=50, max_token_len=20)

# tokens = dev.T

# for x in dev.T:
#   print(x)

