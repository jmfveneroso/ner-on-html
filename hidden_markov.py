import sys
import numpy as np

states = np.array(['H', 'C'])
features = np.array(['1', '2', '3'])

start = [.5, .5]
end   = [1, 1]

transition_mat = np.array([
  [.6, .4],
  [.4, .6]
])

emission_mat = np.array([
  [.2, .5],
  [.4, .4],
  [.4, .1]
])

observations = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])

def forward_algorithm(observations):
  global transition_mat, transition_mat
  alphas = np.zeros((len(observations), len(states)))

  alpha = None
  for i, o in enumerate(observations):
    idx = o - 1
    if alpha is None:
      alpha = start * emission_mat[idx]
      alphas[i] = alpha
      continue

    alpha = np.matmul(alpha, transition_mat)
    alpha *= emission_mat[idx]
    alphas[i] = alpha
  return alphas, np.matmul(alpha, np.transpose(end))

def backward_algorithm(observations):
  global transition_mat, transition_mat
  betas = np.zeros((len(observations), len(states)))

  beta = end
  betas[len(observations) - 1] = beta
  for i, o in enumerate(reversed(observations)):
    idx = o - 1
    beta *= emission_mat[idx]
    if i < len(observations) - 1:
      beta = np.matmul(beta, np.transpose(transition_mat))
      betas[len(observations) - i - 2] = beta
  return betas, np.matmul(beta, np.transpose(start))

def viterbi(observations):
  global states
  backprobs = np.zeros((len(observations), len(states)))
  backpointers = np.zeros((len(observations), len(states)))

  alpha = None
  for i, o in enumerate(observations):
    idx = o - 1
    if alpha is None:
      alpha = start * emission_mat[idx]
      backprobs[i] = alpha
      continue

    alpha_mat = transition_mat * emission_mat[idx]
    alpha_mat = np.transpose(alpha_mat) * alpha
    alpha = np.amax(alpha_mat, axis=1)
    pointers = np.argmax(alpha_mat, axis=1)

    backprobs[i] = np.amax(alpha_mat, axis=1)
    backpointers[i] = np.argmax(alpha_mat, axis=1)

  last_state = np.argmax(backprobs[len(observations) - 1])
  res = [states[last_state]]
  for i in range(0, len(observations) - 1):
    last_state = int(backpointers[len(observations) - i - 1][last_state])
    # print last_state
    res.append(states[last_state])
    
  # print backprobs
  # print backpointers
  res.reverse()
  return res

# print forward_algorithm(observations)

def forward_backward_algorithm(observations):
  global emission_mat, transition_mat
  for i in range(0, 100):
    alphas, n = forward_algorithm(observations)
    betas,  n = backward_algorithm(observations)

    # Transition probs.
    numerator = np.matmul(np.transpose(alphas), betas) * transition_mat
    denominator = np.sum(numerator, axis=1)
    new_transition_probs = (numerator.T / denominator).T

    # Emission probs.
    unary = np.zeros((len(observations), len(features)))
    for i, o in enumerate(observations):
      idx = o - 1
      unary[i][idx] = 1

    numerator = alphas.T * betas.T
    denominator = np.sum(numerator, axis=1)
    numerator = np.matmul(numerator, unary)
    new_emission_probs = numerator.T / denominator

    # print np.round(transition_mat, 4)
    # print np.round(new_transition_probs, 4)
    # print np.round(emission_mat, 4)
    # print np.round(new_emission_probs, 4)

    transition_mat = new_transition_probs
    emission_mat = new_emission_probs
  print np.round(transition_mat, 4)
  print np.round(emission_mat, 4)

forward_backward_algorithm(observations)
print [str(x) for x in observations]
print viterbi(observations)
