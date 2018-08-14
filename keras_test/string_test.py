import keras
import np
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, SimpleRNN, TimeDistributed, GRU, Bidirectional, LSTM
from ChainCRF import ChainCRF
from keras.datasets import mnist
from keras.utils import np_utils
from keras import optimizers

X_ = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0])
Y_ = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

X = np_utils.to_categorical(X_, 3)
Y = np_utils.to_categorical(Y_, 2)


# Softmax regression.

print('Softmax regression')
x = Input(shape=(3, ), dtype='float32') 
y = Dense(2, activation='softmax')(x)

model = Model(inputs=x, outputs=y)
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(x=X, y=Y, batch_size=1, epochs=10)
predictions = model.predict(x=X)

print('Expected:', Y_)
print('Actual:  ', predictions.argmax(axis=-1))


# RNN.

print('RNN')
X = np.expand_dims(X, axis=0)
Y = np.expand_dims(Y, axis=0)

x = Input(shape=(21, 3), dtype='float32') 
rnn = SimpleRNN(5, return_sequences=True)(x)
y = Dense(2, activation='softmax')(rnn)

model = Model(inputs=x, outputs=y)
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(x=X, y=Y, epochs=15)
predictions = model.predict(x=X)
print('Expected:', Y_)
print('Actual:  ', predictions.argmax(axis=-1)[0])


# Bidirectional RNN.

print('Bidirectional RNN')
x = Input(shape=(21, 3), dtype='float32') 
rnn = Bidirectional(SimpleRNN(5, return_sequences=True))(x)
y = Dense(2, activation='softmax')(rnn)

model = Model(inputs=x, outputs=y)
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(x=X, y=Y, epochs=15)
predictions = model.predict(x=X)
print('Expected:', Y_)
print('Actual:  ', predictions.argmax(axis=-1)[0])


# Bidirectional LSTM.

print('Bidirectional LSTM')

x = Input(shape=(21, 3), dtype='float32') 
rnn = Bidirectional(LSTM(5, return_sequences=True))(x)
y = TimeDistributed(Dense(2, activation='softmax'))(rnn)

model = Model(inputs=x, outputs=y)
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(x=X, y=Y, epochs=15)
predictions = model.predict(x=X)
print('Expected:', Y_)
print('Actual:  ', predictions.argmax(axis=-1)[0])


# CRF.

print('Conditional Random Fields')

x = Input(shape=(21, 3), dtype='float32') 
y = TimeDistributed(Dense(2, activation=None))(x)
crf = ChainCRF()
y = crf(y)

model = Model(inputs=x, outputs=y)
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=crf.loss, optimizer=sgd)

model.fit(x=X, y=Y, epochs=15)
predictions = model.predict(x=X)
print('Expected:', Y_)
print('Actual:  ', predictions.argmax(axis=-1)[0])


# Bidirectional LSTM-CRF.

print('Bidirectional LSTM-CRF')

x = Input(shape=(21, 3), dtype='float32') 
rnn = Bidirectional(LSTM(5, return_sequences=True))(x)
y = TimeDistributed(Dense(2, activation=None))(rnn)
crf = ChainCRF()
y = crf(y)

model = Model(inputs=x, outputs=y)
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=crf.loss, optimizer=sgd)

model.fit(x=X, y=Y, epochs=15)
predictions = model.predict(x=X)
print('Expected:', Y_)
print('Actual:  ', predictions.argmax(axis=-1)[0])
