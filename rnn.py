import sys
import math
import numpy as np

states = np.array(['H', 'C'])
features = np.array(['1', '2', '3'])

labels       = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
observations = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])

t = len(observations)
n_x = 3
n_y = 2
n_a = 2
np.random.seed(2)
Waa = np.random.randn(n_a, n_a)
Wax = np.random.randn(n_a, n_x)
Wya = np.random.randn(n_y, n_a)
ba  = np.random.randn(n_a, 1  )
by  = np.random.randn(n_y, 1  )
caches = []

def one_hot(target, num_classes):
  v = np.zeros(num_classes)
  v[target] = 1
  return v

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))

def create_dataset():
  X = np.zeros((n_x, t))
  Y = np.zeros((n_y, t))
  for i in range(t):
    X[:,i] = one_hot(observations[i] - 1, n_x)
    Y[:,i] = one_hot(labels[i], n_y)
  return X, Y

def rnn_cell_forward(xt, a_prev):
  a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
  yt_pred = softmax(np.dot(Wya, a_next) + by)

  cache = (a_prev, a_next, xt) # For backpropagation.
  return a_next, yt_pred, cache

def rnn_forward(x):
  a_next = np.zeros((n_a, 1))
  y_pred = np.zeros((n_y, t))
  a = np.zeros((n_a, t))
  caches = []

  for i in range(t):
    xt = x[:,i,np.newaxis]
    a_next, yt_pred, cache = rnn_cell_forward(xt, a_next)
    y_pred[:,i] = np.ravel(yt_pred)
    a[:,i] = np.ravel(a_next)
    caches.append(cache)

  return a, y_pred, caches

def rnn_cell_backward(da_next, cache):
  a_prev, a_next, xt = cache

  dtanh    = da_next * (1 - a_next**2)
  dbat     = dtanh
  dWaxt    = np.dot(dtanh, xt.T)
  dWaat    = np.dot(dtanh, a_prev.T)
  da_prevt = np.dot(Waa.T, dtanh) 
  return dbat, dWaxt, dWaat, da_prevt

def rnn_backward(da, caches):
  dWax = np.zeros((n_a, n_x))
  dWaa = np.zeros((n_a, n_a))
  dba  = np.zeros((n_a, 1))

  da_prev = np.zeros((n_a, 1))
  for i in reversed(range(t)):
    dbat, dWaxt, dWaat, da_prevt = rnn_cell_backward(da_prev + da[:,i,np.newaxis], caches[i])
    da_prev = da_prevt
    dba  += dbat 
    dWax += dWaxt
    dWaa += dWaat
   
  return dWax, dWaa, dba

def fit(X, Y, learning_rate=0.01):
  global Wax, Waa, Wya, ba, by

  X, Y = create_dataset()

  # Run for 50 epochs
  y_pred = None
  epochs = 50
  decay = learning_rate / epochs
  for i in range(0, epochs):
    learning_rate *= (1. / (1. + decay * i))
    a, y_pred, caches = rnn_forward(X)

    # Cross entropy loss.
    loss = -np.sum(Y * np.log(y_pred))
    predicted_labels = np.argmax(y_pred, axis=0)
    accuracy = 0
    for i in range(len(labels)):
      if predicted_labels[i] == labels[i]:
        accuracy += 1.0
    accuracy = accuracy / len(labels)
    print('Cross entropy loss: ' + str(loss) + ', Accuracy: ' + str(accuracy))

    da = np.zeros((n_a, t))
    dWya = np.zeros((n_y, n_a))
    dby  = np.zeros((n_y, 1))
    for i in range(t):
      dsoftmax = -Y[:,i,np.newaxis] * (1 - y_pred[:,i,np.newaxis])
      dbyt     = dsoftmax
      dWyat    = np.dot(dsoftmax, a[:,i,np.newaxis].T)
      da[:,i]  = np.ravel(np.dot(Wya.T, dsoftmax))
      dby  += dbyt
      dWya += dWyat

    dWax, dWaa, dba = rnn_backward(da, caches)

    # Update weights.
    Wax -= learning_rate * np.clip(dWax, -5.0, 5.0, dWax)
    Waa -= learning_rate * np.clip(dWaa, -5.0, 5.0, dWaa)
    Wya -= learning_rate * np.clip(dWya, -5.0, 5.0, dWya)
    ba  -= learning_rate * np.clip(dba, -5.0, 5.0, dba)
    by  -= learning_rate * np.clip(dby, -5.0, 5.0, dby)
  return y_pred

X, Y = create_dataset()
y_pred = fit(X, Y)
print np.argmax(y_pred, axis=0)
print labels



