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
np.random.seed(1)

Wf = np.random.randn(n_a, n_a + n_x)
Wu = np.random.randn(n_a, n_a + n_x)
Wc = np.random.randn(n_a, n_a + n_x)
Wo = np.random.randn(n_a, n_a + n_x)
Wy = np.random.randn(n_y, n_a)
bf  = np.random.randn(n_a, 1  )
bu  = np.random.randn(n_a, 1  )
bc  = np.random.randn(n_a, 1  )
bo  = np.random.randn(n_a, 1  )
by  = np.random.randn(n_y, 1  )
caches = []

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
  X = np.zeros((n_x, t))
  Y = np.zeros((n_y, t))
  for i in range(t):
    X[:,i] = one_hot(observations[i] - 1, n_x)
    Y[:,i] = one_hot(labels[i], n_y)
  return X, Y

def lstm_cell_forward(xt, a_prev, c_prev):
  concat = np.zeros((n_a + n_x, 1))
  concat[:n_a,] = a_prev
  concat[n_a:,] = xt

  ft      = softmax(np.dot(Wf, concat) + bf)
  ut      = softmax(np.dot(Wu, concat) + bu)
  cct     = np.tanh(np.dot(Wc, concat) + bc)
  c_next  = ft * c_prev + ut * cct
  ot      = softmax(np.dot(Wo, concat) + bo)
  a_next  = ot * np.tanh(c_next)

  yt_pred = softmax(np.dot(Wy, a_next) + by)

  cache = (a_next, c_next, a_prev, c_prev, ft, ut, cct, ot, xt)
  return a_next, c_next, yt_pred, cache

def lstm_forward(x):
  a_next = np.zeros((n_a, 1))
  c_next = np.zeros((n_a, 1))
  y_pred = np.zeros((n_y, t))
  a      = np.zeros((n_a, t))
  caches = []

  for i in range(t):
    xt = x[:,i,np.newaxis]
    a_next, c_next, yt_pred, cache = lstm_cell_forward(xt, a_next, c_next)
    y_pred[:,i] = np.ravel(yt_pred)
    a[:,i] = np.ravel(a_next)
    caches.append(cache)

  return a, y_pred, caches

def lstm_cell_backward(da_next, dc_next, cache):
  (a_next, c_next, a_prev, c_prev, ft, ut, cct, ot, xt) = cache

  dot = ot * (1 - ot)
  dcct = 1 - cct**2
  dut = ut * (1 - ut)
  dft = ft * (1 - ft)

  dut = dc_next * cct + ot * (1 - np.tanh(c_next)**2) * cct * da_next * dut
  dft = dc_next * c_prev + ot * (1 - np.tanh(c_next)**2) * c_prev * da_next * dft
  dot = da_next * np.tanh(c_next) * dot
  dcct = dc_next * ut + ot * (1 - np.tanh(c_next)**2) * ut * da_next * cct * dcct

  concat = np.zeros((n_a + n_x, 1))
  concat[:n_a,] = a_prev
  concat[n_a:,] = xt

  dWf = np.dot(dft, concat.T)
  dWu = np.dot(dut, concat.T)
  dWc = np.dot(dcct, concat.T)
  dWo = np.dot(dot, concat.T)
  dbf = np.sum(dft, keepdims=True, axis=1)
  dbu = np.sum(dut, keepdims=True, axis=1)
  dbc = np.sum(dcct, keepdims=True, axis=1)
  dbo = np.sum(dot, keepdims=True, axis=1)

  da_prev = Wf[:,:n_a].T.dot(dft) + Wu[:,:n_a].T.dot(dut) + Wc[:,:n_a].T.dot(dcct) + Wo[:,:n_a].T.dot(dot)
  dc_prev = dc_next * ft + ot * (1 - np.tanh(c_next)**2) * ft * da_next

  return da_prev, dc_prev, dWf, dWu, dWc, dWo, dbf, dbu, dbc, dbo

def lstm_backward(da, caches):
  dWax = np.zeros((n_a, n_x))
  dWaa = np.zeros((n_a, n_a))
  dba  = np.zeros((n_a, 1))

  dWf = np.zeros((n_a, n_a + n_x))
  dWu = np.zeros((n_a, n_a + n_x))
  dWc = np.zeros((n_a, n_a + n_x))
  dWo = np.zeros((n_a, n_a + n_x))
  dbf = np.zeros((n_a, 1))
  dbu = np.zeros((n_a, 1))
  dbc = np.zeros((n_a, 1))
  dbo = np.zeros((n_a, 1))

  da_prev = np.zeros((n_a, 1))
  dc_prev = np.zeros((n_a, 1))
  for i in reversed(range(t)):
    da_prev, dc_prev, dWft, dWut, dWct, dWot, dbft, dbut, dbct, dbot = lstm_cell_backward(da_prev + da[:,i,np.newaxis], dc_prev, caches[i])

  dWf += dWft 
  dWu += dWut
  dWc += dWct
  dWo += dWot
  dbf += dbft
  dbu += dbut
  dbc += dbct
  dbo += dbot
   
  return dWf, dWu, dWc, dWo, dbf, dbu, dbc, dbo

def fit(X, Y, learning_rate=0.4):
  global Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by

  X, Y = create_dataset()

  # Run for 50 epochs
  y_pred = None
  epochs = 30
  decay = learning_rate / epochs
  for i in range(0, epochs):
    # learning_rate *= (1. / (1. + decay * i))
    a, y_pred, caches = lstm_forward(X)

    # Cross entropy loss.
    loss = -np.sum(Y * np.log(y_pred))
    predicted_labels = np.argmax(y_pred, axis=0)
    accuracy = 0
    for i in range(len(labels)):
      if predicted_labels[i] == labels[i]:
        accuracy += 1.0
    accuracy = accuracy / len(labels)
    print('Cross entropy loss: ' + str(loss) + ', Accuracy: ' + str(accuracy))

    da  = np.zeros((n_a, t))
    dWy = np.zeros((n_y, n_a))
    dby = np.zeros((n_y, 1))
    for i in range(t):
      dsoftmax = -Y[:,i,np.newaxis] * (1 - y_pred[:,i,np.newaxis])
      dbyt     = dsoftmax
      dWyt     = np.dot(dsoftmax, a[:,i,np.newaxis].T)
      da[:,i]  = np.ravel(np.dot(Wy.T, dsoftmax))
      dby  += dbyt
      dWy  += dWyt

    dWf, dWu, dWc, dWo, dbf, dbu, dbc, dbo = lstm_backward(da, caches)

    # Update weights.
    Wf -= learning_rate * np.clip(dWf, -5.0, 5.0, dWu)
    Wu -= learning_rate * np.clip(dWu, -5.0, 5.0, dWu)
    Wc -= learning_rate * np.clip(dWc, -5.0, 5.0, dWc)
    Wo -= learning_rate * np.clip(dWo, -5.0, 5.0, dWo)
    Wy -= learning_rate * np.clip(dWy, -5.0, 5.0, dWy)
    bf  -= learning_rate * np.clip(dbf, -5.0, 5.0, dbf)
    bu  -= learning_rate * np.clip(dbu, -5.0, 5.0, dbu)
    bc  -= learning_rate * np.clip(dbc, -5.0, 5.0, dbc)
    bo  -= learning_rate * np.clip(dbo, -5.0, 5.0, dbo)
    by  -= learning_rate * np.clip(dby, -5.0, 5.0, dby)
  return y_pred

X, Y = create_dataset()
y_pred = fit(X, Y)
print np.argmax(y_pred, axis=0)
print labels
