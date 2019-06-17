import numpy as np
import re

num_labels = 3

def load_raw_dataset(f):
  with open(f, 'r', encoding="utf-8") as f:
    data = f.read().strip()

    sentences = data.split('\n\n')
    sentences = [s for s in sentences if not s.startswith('-DOCSTART-')]
    X = [[t.split(' ') for t in s.split('\n') if len(s) > 0] for s in sentences]
    Y = []
    T = []
    for i, s in enumerate(X):
      tkns, labels = [], []
      for j, t in enumerate(s):
        l = ['O', 'B-PER', 'I-PER'].index(t[1])
        # l = ['O', 'B-PER', 'I-PER'].index(t[3])
        labels.append(l)
        tkns.append(t[0])
        X[i][j] = [X[i][j][0]] + X[i][j][2:]
        # X[i][j] = [X[i][j][0], str(X[i][j][0]).lower()]

      Y.append(labels)
      T.append(tkns)

    return X, Y, T

def is_punctuation(text):
  return re.match("^[\,\;\:\-\"\(\)“”；]$", text)

class HiddenMarkov:
  def __init__(self, timesteps, use_features=True, self_train=False):
    self.naive_bayes = timesteps == 0
    self.time_steps = 1 if (timesteps == 0) else timesteps
    self.use_features = use_features
    self.do_self_train = self_train

    self.num_labels   = 3
    self.num_features = 11
    # self.num_features = 1
    self.num_secondary_features = 2
    self.num_all_features = self.num_features + self.num_secondary_features
    self.num_states = self.num_labels ** self.time_steps
    self.transition_mat = np.ones((self.num_states, self.num_labels))
    self.start = np.zeros((self.num_states, 1))
    self.start[0,:] = 1 # All previous states are label O ("other").
    self.end = np.ones((self.num_states, 1)) # All ending states are equally probable.

    self.feature_counts = []
    for i in range(self.num_all_features):
      self.feature_counts.append([])
      for j in range(self.num_labels):
        self.feature_counts[i].append({'$UNK': 1})

  def idx_to_states(self, idx):
    states = []
    multiplier = self.num_labels ** (self.time_steps - 1)
    for i in range(self.time_steps):
      states.append(int(idx) // int(multiplier))
      idx %= multiplier
      multiplier /= self.num_labels
    return states 
  
  def states_to_idx(self, states):
    if len(states) < self.time_steps:
      raise Exception('Wrong states length.')
  
    acc = 0
    multiplier = 1
    for s in reversed(states):
      acc += int(multiplier) * int(s)
      multiplier *= self.num_labels
    return acc

  def train_features(self, X, Y, which_features=[]):
    if len(which_features) != self.num_all_features:
      which_features = [0] * self.num_all_features

    label_count = np.ones((self.num_labels))
    for i in range(len(Y)):
      for j in range(len(Y[i])):
        label_count += Y[i][j]
        y = Y[i][j]
  
        f = X[i][j][1:1+self.num_all_features]
        for k in range(self.num_all_features):
          if which_features[k] == 0:
            continue

          key = ''
          if k < len(f):
            key = f[k]

          if not key in self.feature_counts[k][y]:
            self.feature_counts[k][y][key] = 0
          self.feature_counts[k][y][key] += 1
 
    # Consolidate vocabularies. 
    feature_maps = []
    for i in range(self.num_all_features):
      feature_maps.append({})
      for j in range(self.num_labels):
        for k in self.feature_counts[i][j]:
          feature_maps[i][k] = True

    for i in range(self.num_all_features):
      if which_features[i] == 0:
        continue

      for j in range(self.num_labels):
        for k in feature_maps[i]:
          if not k in self.feature_counts[i][j]:
            self.feature_counts[i][j][k] = 1

    for i in range(self.num_all_features):
      if which_features[i] == 0:
        continue

      for j in range(self.num_labels):
        total_count = sum([self.feature_counts[i][j][k] for k in self.feature_counts[i][j]])
        for k in self.feature_counts[i][j]:
          self.feature_counts[i][j][k] /= float(total_count)

  def train_transitions(self, X, Y):
    for i in range(len(Y)):
      states = [0] * self.time_steps
      for j in range(len(Y[i])):
        y = Y[i][j]
        idx = self.states_to_idx(states)

        self.transition_mat[idx,y] += 1
        states.pop(0) 
        states.append(y) 
  
    # self.transition_mat /= np.expand_dims(np.sum(self.transition_mat, axis=1), axis=1)
    # self.transition_mat = np.nan_to_num(self.transition_mat)

    if self.naive_bayes:
      self.transition_mat = np.sum(self.transition_mat, axis=0)
      self.transition_mat /= np.sum(self.transition_mat)
    else:
      self.transition_mat /= np.expand_dims(np.sum(self.transition_mat, axis=1), axis=1)
      self.transition_mat = np.nan_to_num(self.transition_mat)

  def fit(self, X, Y):
    which_features = [0] * self.num_all_features 
    for i in range(0,1):
      which_features[i] = 1

    if self.use_features:
      for i in range(1,1+self.num_features):
        which_features[i] = 1
        # if i != 1 and i != 2:
        #   which_features[i] = 0
      which_features[3] = 0
      which_features[4] = 0
      which_features[1] = 1
      which_features[2] = 1
      which_features[8] = 1
      which_features[9] = 1

    self.train_features(X, Y, which_features)
    self.train_transitions(X, Y)

  def viterbi(self, X):
    pointers = np.zeros((len(X), self.num_states), dtype=int)
  
    state_probs = self.start 
    for i in range(len(X)):
      emission = np.ones(self.num_labels)
  
      f = X[i][1:1+self.num_features+self.num_secondary_features]
      for k in range(self.num_features+self.num_secondary_features):
        for y in range(self.num_labels):
          key = ''
          if k < len(f):
            key = f[k]

          if key in self.feature_counts[k][y]: 
            emission[y] *= self.feature_counts[k][y][key]
          else:
            emission[y] *= self.feature_counts[k][y]['$UNK']
      emission[emission == 1] = 0

      p = state_probs * self.transition_mat * emission

      state_probs = np.zeros((self.num_states, 1))
      for s in range(self.num_states):
        for l in range(self.num_labels):
          states = self.idx_to_states(s)
          states.pop(0)
          states.append(l)
          idx = self.states_to_idx(states)

          if p[s,l] > state_probs[idx,0]:
            pointers[i,idx] = s
            state_probs[idx,0] = p[s,l]

    idx = np.argmax(state_probs)
    labels = [] 
    for i in reversed(range(len(X))):
      states = self.idx_to_states(idx)
      labels.append(states[-1])
      idx = pointers[i,idx]
    labels = list(reversed(labels))

    return labels

  def nb_predict(self, X):
    labels = [] 
    for i in range(len(X)):
      emission = np.ones(self.num_labels)
  
      f = X[i][1:1+self.num_features+self.num_secondary_features]
      for k in range(self.num_features+self.num_secondary_features):
        for y in range(self.num_labels):
          key = ''
          if k < len(f):
            key = f[k]

          if key in self.feature_counts[k][y]: 
            emission[y] *= self.feature_counts[k][y][key]
          else:
            emission[y] *= self.feature_counts[k][y]['$UNK']
      # emission[emission == 1] = 0

      print(X[i][0])
      print(emission)
      p = self.transition_mat * emission
      print(p)
      labels.append(p.argmax())
      print(p.argmax())
    return labels

  def get_predictions(self, X):
    y = []
    for i in range(len(X)):
      if self.naive_bayes:
        labels = self.nb_predict(X[i])
      else:
        labels = self.viterbi(X[i])
      y.append(labels) 
    return y

  def self_train(self, X):
    # Reset secondary features from previous runs.
    for i in range(self.num_features, self.num_all_features):
      self.feature_counts[i] = []
      for j in range(self.num_labels):
        self.feature_counts[i].append({'$UNK': 1})

    Y = self.get_predictions(X)
    which_features = [0] * self.num_features 
    which_features += [1] * self.num_secondary_features
    # which_features[-2] = 0
    self.train_features(X, Y, which_features)

  def predict(self, X):
    if self.do_self_train:
      self.self_train(X)
    return self.get_predictions(X)
