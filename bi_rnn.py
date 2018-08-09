import sys
import math
import numpy as np

labels       = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
observations = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])
t = len(observations)

def one_hot(target, num_classes):
  v = np.zeros(num_classes)
  v[target] = 1
  return v

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))

def create_dataset():
  X = np.zeros((3, t))
  Y = np.zeros((2, t))
  for i in range(t):
    X[:,i] = one_hot(observations[i] - 1, 3)
    Y[:,i] = one_hot(labels[i], 2)
  return X, Y

class RNN():
  def __init__(self, n_x, n_y, n_a, t):
    self.n_x = n_x
    self.n_y = n_y
    self.n_a = n_a
    self.t   = t

    np.random.seed(5)
    self.Waa = np.random.randn(n_a, n_a)
    self.Wax = np.random.randn(n_a, n_x)
    self.ba  = np.random.randn(n_a, 1  )

  def rnn_cell_forward(self, xt, a_prev):
    a_next = np.tanh(np.dot(self.Wax, xt) + np.dot(self.Waa, a_prev) + self.ba)
  
    cache = (a_prev, a_next, xt) # For backpropagation.
    return a_next, cache

  def rnn_forward(self, x):
    a_next = np.zeros((self.n_a, 1))
    a = np.zeros((self.n_a, self.t))
    caches = []
  
    for i in range(t):
      xt = x[:,i,np.newaxis]
      a_next, cache = self.rnn_cell_forward(xt, a_next)
      a[:,i] = np.ravel(a_next)
      caches.append(cache)
  
    # return a, y_pred, caches
    return a, caches

  def rnn_cell_backward(self, da_next, cache):
    a_prev, a_next, xt = cache
  
    dtanh    = da_next * (1 - a_next**2)
    dbat     = dtanh
    dWaxt    = np.dot(dtanh, xt.T)
    dWaat    = np.dot(dtanh, a_prev.T)
    da_prevt = np.dot(self.Waa.T, dtanh) 
    return dbat, dWaxt, dWaat, da_prevt

  def rnn_backward(self, da, caches, learning_rate):
    dWax = np.zeros((self.n_a, self.n_x))
    dWaa = np.zeros((self.n_a, self.n_a))
    dba  = np.zeros((self.n_a, 1))
  
    da_prev = np.zeros((self.n_a, 1))
    for i in reversed(range(t)):
      dbat, dWaxt, dWaat, da_prevt = self.rnn_cell_backward(da_prev + da[:,i,np.newaxis], caches[i])
      da_prev = da_prevt
      dba  += dbat 
      dWax += dWaxt
      dWaa += dWaat
     
    self.Wax -= learning_rate * np.clip(dWax, -5.0, 5.0, dWax)
    self.Waa -= learning_rate * np.clip(dWaa, -5.0, 5.0, dWaa)
    self.ba  -= learning_rate * np.clip(dba, -5.0, 5.0, dba)
    return dWax, dWaa, dba

class BiRNN():
  def __init__(self, n_x, n_y, n_a, t):
    self.n_x = n_x
    self.n_y = n_y
    self.n_a = n_a
    self.t   = t
    np.random.seed(3)
    self.Wya = np.random.randn(n_y, n_a + n_a)
    self.by  = np.random.randn(n_y, 1  )
    self.f_rnn = RNN(n_x, n_y, n_a, t)
    self.b_rnn = RNN(n_x, n_y, n_a, t)

    self.caches1 = None
    self.caches2 = None
    self.concat  = None
    self.X = None
    self.y_pred = None

  def set_input(self, X):
    self.X = X

  def propagate(self):
    a1, self.caches1 = self.f_rnn.rnn_forward(self.X)
    a2, self.caches2 = self.b_rnn.rnn_forward(np.flip(self.X, 1))

    self.concat = np.zeros((self.n_a + self.n_a, self.t))
    self.concat[:self.n_a,:] = a1
    self.concat[self.n_a:,:] = a2

    self.y_pred = softmax(np.dot(self.Wya, self.concat) + self.by)
    return self.y_pred

  def backpropagate(self, dLoss, learning_rate=0.001):
    da = np.zeros((self.n_a + self.n_a, self.t))
    dWya = np.zeros((self.n_y, self.n_a + self.n_a))
    dby  = np.zeros((self.n_y, 1))
    for i in range(t):
      dsoftmax = dLoss[:,i,np.newaxis] * self.y_pred[:,i,np.newaxis] * (1 - self.y_pred[:,i,np.newaxis])
      dbyt     = dsoftmax
      dWyat    = np.dot(dsoftmax, self.concat[:,i,np.newaxis].T)
      da[:,i]  = np.ravel(np.dot(self.Wya.T, dsoftmax))
      dby  += dbyt
      dWya += dWyat
  
    self.f_rnn.rnn_backward(da[:self.n_a,], self.caches1, learning_rate)
    self.b_rnn.rnn_backward(np.flip(da[self.n_a:,], 1), self.caches2, learning_rate)
  
    # Update weights.
    self.Wya -= learning_rate * np.clip(dWya, -5.0, 5.0, dWya)
    self.by  -= learning_rate * np.clip(dby, -5.0, 5.0, dby)

  def fit(self, X, Y, learning_rate=0.001, epochs=30):
    decay = learning_rate / epochs
    for i in range(0, epochs):
      learning_rate *= (1. / (1. + decay * i))

      y_pred = self.propagate()
      self.backpropagate(-Y / y_pred, learning_rate)

      # Cross entropy loss.
      loss = -np.sum(Y * np.log(y_pred))
      predicted_labels = np.argmax(y_pred, axis=0)
      accuracy = 0
      for i in range(len(labels)):
        if predicted_labels[i] == labels[i]:
          accuracy += 1.0
      accuracy = accuracy / len(labels)
      print('Cross entropy loss: ' + str(loss) + ', Accuracy: ' + str(accuracy))
    return y_pred

if __name__ == "__main__":
  X, Y = create_dataset()
  rnn = BiRNN(3, 2, 2, t)
  rnn.set_input(X)
  y_pred = rnn.fit(X, Y)
  print np.argmax(y_pred, axis=0)
  print labels
