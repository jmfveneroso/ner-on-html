import numpy as np
import matplotlib.pyplot as plt
import math
np.set_printoptions(threshold=np.inf)

learning_rate = 0.000000001
data = []
w = np.full((10, 785), 0.00001, dtype=float)

def load_data():
  with open('mnist.data') as f:
    for line in f:
      arr = line.strip().split(',')
      features = np.concatenate([[1], np.asarray(arr[1:], dtype=int)])
      data.append((features, int(arr[0])))

def softmax(x):
  # Numerically stable.
  logit = np.matmul(w, x)
  shift_x = logit - np.max(logit)
  exps = np.exp(shift_x)
  return exps / np.sum(exps)

def fit(epochs):
  global w, data
  for i in range(0, epochs):
    neg_log_likelihood = 0
    correct = 0
  
    w_update = np.full((10, 785), 0, dtype=float)
    for d in data:
      expected_class = d[1]
      probs = softmax(d[0])
      if np.argmax(probs) == expected_class:
        correct += 1
  
      neg_log_likelihood += -math.log(probs[expected_class])
      derivative = probs[expected_class] - 1
      w_update[expected_class] += d[0] * (-derivative * learning_rate)
    w = np.add(w, w_update)
    print 'Epoch:', i+1, '; Accuracy:', float(correct) / len(data), '; Negative log likelihood:', float(neg_log_likelihood) / len(data)

load_data()
fit(10000)
