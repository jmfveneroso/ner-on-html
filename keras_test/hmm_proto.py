import np
from util import load_raw_dataset
from tokenizer import remove_accents, tokenize_text
from metric import evaluate, get_names, one_hot_to_labels

class HiddenMarkov:
  def __init__(self, timesteps, use_gazetteer=True, use_features=True, self_train=False):
    self.time_steps   = timesteps
    self.use_gazetteer = use_gazetteer
    self.use_features = use_features
    self.do_self_train = self_train

    self.num_labels   = 4
    self.num_features = 10
    self.num_secondary_features = 5
    self.num_all_features = self.num_features + self.num_secondary_features
    self.num_states = self.num_labels ** self.time_steps
    self.transition_mat = np.zeros((self.num_states, self.num_labels))
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
      for j in range(50):
        if np.sum(Y[i,j]) == 0:
          break
  
        label_count += Y[i,j]
        y = np.where(Y[i,j] == 1)[0][0]
  
        f = X[i][j][1:1+self.num_all_features]
        for k in range(self.num_all_features):
          if which_features[k] == 0:
            continue

          key = f[k]
          if not key in self.feature_counts[k][y]:
            self.feature_counts[k][y][key] = 0
          self.feature_counts[k][y][key] += 1
  
    for i in range(self.num_all_features):
      if which_features[i] == 0:
        continue

      for j in range(self.num_labels):
        total_count = sum([self.feature_counts[i][j][k] for k in self.feature_counts[i][j]])
        for k in self.feature_counts[i][j]:
          self.feature_counts[i][j][k] /= float(total_count)

  def train_transitions(self, X, Y):
    states = [0] * self.time_steps
    for i in range(len(Y)):
      for j in range(50):
        if np.sum(Y[i,j]) == 0:
          break
  
        y = np.where(Y[i,j] == 1)[0][0]
        idx = self.states_to_idx(states)
        self.transition_mat[idx,y] += 1
        states.pop(0) 
        states.append(y) 
  
    self.transition_mat /= np.expand_dims(np.sum(self.transition_mat, axis=1), axis=1)
    self.transition_mat = np.nan_to_num(self.transition_mat)

  def fit(self, X, Y):
    which_features = [0] * self.num_all_features 
    for i in range(0,1):
      which_features[i] = 1
    if self.use_gazetteer:
      for i in range(1,3):
        which_features[i] = 1
      for i in range(8,10):
        which_features[i] = 1
    if self.use_features:
      for i in range(3,8):
        which_features[i] = 1

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
          if f[k] in self.feature_counts[k][y]: 
            emission[y] *= self.feature_counts[k][y][f[k]]
          else:
            emission[y] *= self.feature_counts[k][y]['$UNK']
  
      p = state_probs * self.transition_mat * emission
      state_probs = np.zeros((self.num_states, 1))
      for s in range(self.num_states):
        for l in range(self.num_labels):
          states = self.idx_to_states(s)
          states.pop(0)
          states.append(l)
          idx = self.states_to_idx(states)
           
          if p[s,l] > state_probs[idx,:]:
            pointers[i,idx] = s
            state_probs[idx,0] = p[s,l]
  
    idx = np.argmax(state_probs)
    labels = [] 
    for i in reversed(range(len(X))):
      states = self.idx_to_states(idx)
      labels.append(states[-1])
      idx = pointers[i,idx]
    return list(reversed(labels))

  def get_predictions(self, X):
    y = np.zeros((len(X), 50, self.num_labels))
    for i in range(len(X)):
      labels = self.viterbi(X[i])
  
      # To one hot.
      for j, l in enumerate(labels):
        y[i,j,int(l)] = 1
    return y

  def self_train(self, X):
    Y = self.get_predictions(X)
    which_features = [0] * self.num_features 
    which_features += [1] * self.num_secondary_features
    self.train_features(X, Y, which_features)

  def predict(self, X):
    if self.do_self_train:
      self.self_train(X)
    return self.get_predictions(X)

