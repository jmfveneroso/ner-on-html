import keras
import np
from matplotlib import pyplot as plt
from util import Dataset

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, SimpleRNN, TimeDistributed, GRU, Bidirectional, LSTM, Embedding, concatenate, Conv1D, GlobalMaxPooling1D
from ChainCRF import ChainCRF
from keras.datasets import mnist
from keras.utils import np_utils
from keras import optimizers


# Bidirectional LSTM-CRF.
class LstmCrf():
  def __init__(
    self, 
    lstm_cells=30, 
    char_embedding_size=10
  ):
    self.x = []

  # def model():




ds = Dataset()
word_embedding_matrix = ds.load_word_embeddings('')
X1, X2, Y = ds.load_dataset('string_test_data.txt')

num_words = 12 + 1
char_embedding_matrix = np.random.randn(128, 128)


inputs = []

x = Input(shape=(X1.shape[1], X1.shape[2]), dtype='int32')
inputs.append(x)

x = Flatten()(x)
emb = Embedding(
  input_dim=word_embedding_matrix.shape[0], 
  output_dim=word_embedding_matrix.shape[1], 
  weights=[word_embedding_matrix], 
  trainable=False
)(x)

shared_layer = [emb]


# Char embeddings.
x = Input(shape=(X2.shape[1], X2.shape[2]), dtype='int32')
inputs.append(x)
emb2 = TimeDistributed(Embedding(input_dim=128, output_dim=128, weights=[char_embedding_matrix], trainable=True))(x)

# LSTM.
emb2 = TimeDistributed(Bidirectional(LSTM(5, return_sequences=False)))(emb2)

# CNN.
# emb2 = TimeDistributed(Conv1D(3, 3))(emb2)
# emb2 = TimeDistributed(GlobalMaxPooling1D())(emb2)

shared_layer.append(emb2)
emb = concatenate(shared_layer)


rnn = Bidirectional(LSTM(5, return_sequences=True))(emb)
y = TimeDistributed(Dense(Y.shape[1], activation=None))(emb)
crf = ChainCRF()
y = crf(y)

model = Model(inputs=inputs, outputs=y)
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=crf.loss, optimizer=sgd)

model.fit(x=[X1,X2], y=Y, epochs=30)
predictions = model.predict(x=[X1,X2])
print('Expected:', Y.argmax(axis=-1)[0])
print('Actual:  ', predictions.argmax(axis=-1)[0])
print('Expected:', Y.argmax(axis=-1)[1])
print('Actual:  ', predictions.argmax(axis=-1)[1])
