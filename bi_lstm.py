import sys
import math
import numpy as np

states = np.array(['H', 'C'])
features = np.array(['1', '2', '3'])

labels       = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
observations = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])

def one_hot(target, num_classes):
  v = np.zeros(num_classes)
  v[target] = 1
  return v

def softmax(x):
  # Numerically stable.
  shift_x = x - np.max(x)
  exps = np.exp(shift_x)
  res = exps / np.sum(exps, axis=0)
  return res

def create_dataset():
  X = np.zeros((3, len(observations)))
  Y = np.zeros((2, len(observations)))
  for i in range(len(observations)):
    X[:,i] = one_hot(observations[i] - 1, 3)
    Y[:,i] = one_hot(labels[i], 2)
  return X, Y

class LSTM():
  def __init__(self, n_x, n_y, n_a, t):
    self.n_x = n_x
    self.n_y = n_y
    self.n_a = n_a
    self.t   = t

    np.random.seed(1)
    self.Wf = np.random.randn(n_a, n_a + n_x)
    self.Wu = np.random.randn(n_a, n_a + n_x)
    self.Wc = np.random.randn(n_a, n_a + n_x)
    self.Wo = np.random.randn(n_a, n_a + n_x)
    self.Wy = np.random.randn(n_y, n_a)
    self.bf  = np.random.randn(n_a, 1  )
    self.bu  = np.random.randn(n_a, 1  )
    self.bc  = np.random.randn(n_a, 1  )
    self.bo  = np.random.randn(n_a, 1  )
    self.by  = np.random.randn(n_y, 1  )
    self.caches = []

  def lstm_cell_forward(self, xt, a_prev, c_prev):
    concat = np.zeros((self.n_a + self.n_x, 1))
    concat[:self.n_a,] = a_prev
    concat[self.n_a:,] = xt
  
    ft      = softmax(np.dot(self.Wf, concat) + self.bf)
    ut      = softmax(np.dot(self.Wu, concat) + self.bu)
    cct     = np.tanh(np.dot(self.Wc, concat) + self.bc)
    c_next  = ft * c_prev + ut * cct
    ot      = softmax(np.dot(self.Wo, concat) + self.bo)
    a_next  = ot * np.tanh(c_next)
  
    yt_pred = softmax(np.dot(self.Wy, a_next) + self.by)
  
    cache = (a_next, c_next, a_prev, c_prev, ft, ut, cct, ot, xt)
    return a_next, c_next, yt_pred, cache
  
  def lstm_forward(self, x):
    a_next = np.zeros((self.n_a, 1))
    c_next = np.zeros((self.n_a, 1))
    y_pred = np.zeros((self.n_y, self.t))
    a      = np.zeros((self.n_a, self.t))
    caches = []
  
    for i in range(self.t):
      xt = x[:,i,np.newaxis]
      a_next, c_next, yt_pred, cache = self.lstm_cell_forward(xt, a_next, c_next)
      y_pred[:,i] = np.ravel(yt_pred)
      a[:,i] = np.ravel(a_next)
      caches.append(cache)
  
    return a, y_pred, caches
  
  def lstm_cell_backward(self, da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, ut, cct, ot, xt) = cache
  
    dot = ot * (1 - ot)
    dcct = 1 - cct**2
    dut = ut * (1 - ut)
    dft = ft * (1 - ft)
  
    dut = dc_next * cct + ot * (1 - np.tanh(c_next)**2) * cct * da_next * dut
    dft = dc_next * c_prev + ot * (1 - np.tanh(c_next)**2) * c_prev * da_next * dft
    dot = da_next * np.tanh(c_next) * dot
    dcct = dc_next * ut + ot * (1 - np.tanh(c_next)**2) * ut * da_next * cct * dcct
  
    concat = np.zeros((self.n_a + self.n_x, 1))
    concat[:self.n_a,] = a_prev
    concat[self.n_a:,] = xt
  
    dWf = np.dot(dft, concat.T)
    dWu = np.dot(dut, concat.T)
    dWc = np.dot(dcct, concat.T)
    dWo = np.dot(dot, concat.T)
    dbf = np.sum(dft, keepdims=True, axis=1)
    dbu = np.sum(dut, keepdims=True, axis=1)
    dbc = np.sum(dcct, keepdims=True, axis=1)
    dbo = np.sum(dot, keepdims=True, axis=1)
  
    da_prev = self.Wf[:,:self.n_a].T.dot(dft) + self.Wu[:,:self.n_a].T.dot(dut) + self.Wc[:,:self.n_a].T.dot(dcct) + self.Wo[:,:self.n_a].T.dot(dot)
    dc_prev = dc_next * ft + ot * (1 - np.tanh(c_next)**2) * ft * da_next
  
    return da_prev, dc_prev, dWf, dWu, dWc, dWo, dbf, dbu, dbc, dbo
  
  def lstm_backward(self, da, caches):
    dWax = np.zeros((self.n_a, self.n_x))
    dWaa = np.zeros((self.n_a, self.n_a))
    dba  = np.zeros((self.n_a, 1))
  
    dWf = np.zeros((self.n_a, self.n_a + self.n_x))
    dWu = np.zeros((self.n_a, self.n_a + self.n_x))
    dWc = np.zeros((self.n_a, self.n_a + self.n_x))
    dWo = np.zeros((self.n_a, self.n_a + self.n_x))
    dbf = np.zeros((self.n_a, 1))
    dbu = np.zeros((self.n_a, 1))
    dbc = np.zeros((self.n_a, 1))
    dbo = np.zeros((self.n_a, 1))
  
    da_prev = np.zeros((self.n_a, 1))
    dc_prev = np.zeros((self.n_a, 1))
    for i in reversed(range(self.t)):
      da_prev, dc_prev, dWft, dWut, dWct, dWot, dbft, dbut, dbct, dbot = self.lstm_cell_backward(da_prev + da[:,i,np.newaxis], dc_prev, caches[i])
  
    dWf += dWft 
    dWu += dWut
    dWc += dWct
    dWo += dWot
    dbf += dbft
    dbu += dbut
    dbc += dbct
    dbo += dbot
     
    return dWf, dWu, dWc, dWo, dbf, dbu, dbc, dbo

  def fit(self, X, Y, learning_rate=0.4, epochs=30):
    global Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
  
    y_pred = None
    decay = learning_rate / epochs
    for i in range(0, epochs):
      # learning_rate *= (1. / (1. + decay * i))
      a, y_pred, caches = self.lstm_forward(X)
  
      # Cross entropy loss.
      loss = -np.sum(Y * np.log(0.1 + y_pred))
      predicted_labels = np.argmax(y_pred, axis=0)
      accuracy = 0
      for i in range(len(labels)):
        if predicted_labels[i] == labels[i]:
          accuracy += 1.0
      accuracy = accuracy / len(labels)
      print('Cross entropy loss: ' + str(loss) + ', Accuracy: ' + str(accuracy))
  
      da  = np.zeros((self.n_a, self.t))
      dWy = np.zeros((self.n_y, self.n_a))
      dby = np.zeros((self.n_y, 1))
      for i in range(self.t):
        dsoftmax = -Y[:,i,np.newaxis] * (1 - y_pred[:,i,np.newaxis])
        dbyt     = dsoftmax
        dWyt     = np.dot(dsoftmax, a[:,i,np.newaxis].T)
        da[:,i]  = np.ravel(np.dot(self.Wy.T, dsoftmax))
        dby  += dbyt
        dWy  += dWyt
  
      dWf, dWu, dWc, dWo, dbf, dbu, dbc, dbo = self.lstm_backward(da, caches)
  
      # Update weights.
      self.Wf -= learning_rate * np.clip(dWf, -1.0, 1.0, dWu)
      self.Wu -= learning_rate * np.clip(dWu, -1.0, 1.0, dWu)
      self.Wc -= learning_rate * np.clip(dWc, -1.0, 1.0, dWc)
      self.Wo -= learning_rate * np.clip(dWo, -1.0, 1.0, dWo)
      self.Wy -= learning_rate * np.clip(dWy, -1.0, 1.0, dWy)
      self.bf -= learning_rate * np.clip(dbf, -1.0, 1.0, dbf)
      self.bu -= learning_rate * np.clip(dbu, -1.0, 1.0, dbu)
      self.bc -= learning_rate * np.clip(dbc, -1.0, 1.0, dbc)
      self.bo -= learning_rate * np.clip(dbo, -1.0, 1.0, dbo)
      self.by -= learning_rate * np.clip(dby, -1.0, 1.0, dby)
    return y_pred

if __name__ == "__main__":
  lstm = LSTM(3, 2, 2, len(observations))
  X, Y = create_dataset()
  y_pred = lstm.fit(X, Y, learning_rate=0.03, epochs=100000)
  # print np.argmax(y_pred, axis=0)
  # print labels
