import sys
import math
import numpy as np

states = np.array(['H', 'C'])
features = np.array(['1', '2', '3'])

labels       = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
observations = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])

params = np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1])

def get_features(y, prev_y, x):
  return np.array([
    1 if y == 1 and prev_y == 1 and x == 1 else 0, # C C 1
    1 if y == 1 and prev_y == 1 and x == 2 else 0, # C C 2
    1 if y == 1 and prev_y == 1 and x == 3 else 0, # C C 3
    1 if y == 0 and prev_y == 1 and x == 1 else 0, # C H 1
    1 if y == 0 and prev_y == 1 and x == 2 else 0, # C H 2
    1 if y == 0 and prev_y == 1 and x == 3 else 0, # C H 3
    1 if y == 1 and prev_y == 0 and x == 1 else 0, # H C 1
    1 if y == 1 and prev_y == 0 and x == 2 else 0, # H C 2
    1 if y == 1 and prev_y == 0 and x == 3 else 0, # H C 3
    1 if y == 0 and prev_y == 0 and x == 1 else 0, # H H 1
    1 if y == 0 and prev_y == 0 and x == 2 else 0, # H H 2
    1 if y == 0 and prev_y == 0 and x == 3 else 0  # H H 3
  ])

def prob(x):
  probs = np.zeros((len(states), len(states)))
  for prev_y, v1 in enumerate(states):
    for y, v2 in enumerate(states):
      probs[prev_y][y] = np.exp(np.sum(params * get_features(y, prev_y, x)))
  return probs

def forward_algorithm(observations):
  alphas = np.zeros((len(observations), len(states)))

  alpha = [0.5, 0.5]
  for i, o in enumerate(observations):
    alpha = np.matmul(alpha, prob(o))
    alphas[i] = alpha
  return alphas, np.sum(alpha)

def backward_algorithm(observations):
  betas = np.zeros((len(observations), len(states)))

  beta = np.repeat(1, len(states))
  betas[len(observations) - 1] = beta
  for i, o in enumerate(reversed(observations)):
    beta = np.matmul(beta, np.transpose(prob(o)))
    if i < len(observations) - 1:
      betas[len(observations) - i - 2] = beta
  return betas, np.matmul(beta, np.transpose(np.array([0.5, 0.5])))

def viterbi(observations):
  global states
  backprobs = np.zeros((len(observations), len(states)))
  backpointers = np.zeros((len(observations), len(states)))

  alpha = np.array([0.5, 0.5])
  for i, o in enumerate(observations):
    alpha_mat = alpha.reshape((2, 1)) * prob(o)
    alpha = np.amax(alpha_mat.T, axis=1)
    pointers = np.argmax(alpha_mat.T, axis=1)
    backprobs[i] = alpha
    backpointers[i] = pointers

  last_state = np.argmax(backprobs[len(observations) - 1])
  res = [states[last_state]]
  for i in range(0, len(observations) - 1):
    last_state = int(backpointers[len(observations) - i - 1][last_state])
    res.append(states[last_state])
    
  # print backprobs
  # print backpointers
  res.reverse()
  return res

def log_likelihood(observations, partition_value):
  value = 0
  prev_y = 0
  for i, o in enumerate(observations):
    value += np.sum(params * get_features(labels[i], prev_y, o))
    prev_y = labels[i]
  return value - np.log(partition_value)

learning_rate = 0.1
def fit(observations):
  global params, learning_rate

  for i in range(0, 10):
    # Empirical feature count.
    feature_count = np.zeros(len(params))
    prev_y = 0
    for i, o in enumerate(observations):
      feature_count += get_features(labels[i], prev_y, o) 
      prev_y = labels[i]

    alphas, n = forward_algorithm(observations)
    betas,  n = backward_algorithm(observations)

    print 'Log likelihood:', log_likelihood(observations, n)

    # Expected feature count.
    expected_feature_count = np.zeros(len(params))
    for i, o in enumerate(observations):
      mat = np.matmul(alphas[i].reshape((2, 1)), betas[i].reshape((1, 2))) / n

      features = np.zeros(len(params))
      for prev_y, v1 in enumerate(states):
        for y, v2 in enumerate(states):
          features += mat[prev_y][y] * get_features(y, prev_y, o)
      expected_feature_count += features
    # print feature_count
    # print expected_feature_count

    # Update weights.
    derivatives = feature_count - expected_feature_count
    params += learning_rate * derivatives 

fit(observations)
print [states[x] for x in labels]
print viterbi(observations)
