import keras
import np
import re
from util import Dataset, WordEmbeddings
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, SimpleRNN, TimeDistributed, GRU, Bidirectional, LSTM, Embedding, concatenate, Conv1D, GlobalMaxPooling1D
from ChainCRF import ChainCRF
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import Callback
from keras import optimizers
from metric import evaluate, get_names, one_hot_to_labels

class Metrics(Callback):
  def __init__(self, the_model):
    self.max_f_measure = 0
    self.the_model = the_model

  def on_epoch_end(self, epoch, logs={}):
    val_predict, val_targ, T = self.the_model.validate()
    
    a, p, r, f1, c, i, m = evaluate(val_predict, val_targ, T)
    
    print('Accuracy:', a)
    print('Precision:', p)
    print('Recall:', r)
    print('F1:', f1)
    print('Correct:', c)
    print('Incorrect:', i)
    print('Missed:', m)

    if f1 > self.max_f_measure:
      self.max_f_measure = f1
      self.the_model.save()
    else:
      print("No improvement")
   
class LstmCrf():
  def __init__(
    self, 
    name,
    lstm_cells=100, 
    use_gazetteer=True, 
    use_features=True, 
    char_embedding_size=30,
    num_cnn_filters=30,
    num_labels=4,
    char_lstm_size=25,
    model_type=['lstm-crf', 'lstm-crf-cnn', 'lstm-crf-lstm', 'crf'][0],
    dev_dataset=None,
    validate_dataset=None,
    test_dataset=None,
  ):
    self.name = name
    self.lstm_cells = lstm_cells
    self.num_cnn_filters = num_cnn_filters
    self.num_labels = num_labels
    self.char_lstm_size = char_lstm_size
    self.model_type = model_type
    self.dev_dataset = dev_dataset
    self.validate_dataset = validate_dataset
    self.test_dataset = test_dataset
    self.use_gazetteer = use_gazetteer
    self.use_features = use_features

  def create(self, word_embedding_matrix):
    x1_shape = self.dev_dataset.X1.shape
    x2_shape = self.dev_dataset.X2.shape
    x3_shape = self.dev_dataset.X3.shape
    x4_shape = self.dev_dataset.X4.shape

    char_embedding_matrix = np.random.randn(128, 30)

    x = Input(shape=(x1_shape[1], x1_shape[2]), dtype='int32')
    inputs = [x]
    
    # Word embeddings.
    word_emb = Embedding(
      input_dim=word_embedding_matrix.shape[0], 
      output_dim=word_embedding_matrix.shape[1], 
      weights=[word_embedding_matrix], 
      trainable=False
    )(Flatten()(x))

    # Feature embeddings.
    shared_layer = [word_emb]

    if self.use_gazetteer:
      x2 = Input(shape=(x2_shape[1], x2_shape[2]), dtype='float32')
      inputs.append(x2)
      shared_layer.append(x2)

    if self.use_features:
      x3 = Input(shape=(x3_shape[1], x3_shape[2]), dtype='float32')
      inputs.append(x3)
      shared_layer.append(x3)

    # Char embeddings.
    if self.model_type != 'lstm-crf' and self.model_type != 'crf':
      x = Input(shape=(x4_shape[1], x4_shape[2]), dtype='int32')
      inputs.append(x)

      char_emb = TimeDistributed(Embedding(
        input_dim=char_embedding_matrix.shape[0],  
        output_dim=char_embedding_matrix.shape[1],
        weights=[char_embedding_matrix], 
        trainable=True
      ))(x)
      
      # LSTM char embeddings from Lample et al., 2016.
      if self.model_type != 'lstm-crf-lstm':
        char_emb = TimeDistributed(Bidirectional(LSTM(self.char_lstm_size, return_sequences=False)))(char_emb)

      # CNN char embeddings from Ma and Hovy, 2016.
      elif self.model_type != 'lstm-crf-cnn':
        char_emb = TimeDistributed(Conv1D(self.num_cnn_filters, 3))(char_emb)
        char_emb = TimeDistributed(GlobalMaxPooling1D())(char_emb)
      
      shared_layer.append(char_emb)
    shared_layer = concatenate(shared_layer)

    if self.model_type == 'crf':
      y = TimeDistributed(Dense(self.num_labels, activation=None))(shared_layer)
      crf = ChainCRF()
      y = crf(y)
    else:
      lstm = Bidirectional(LSTM(self.lstm_cells, return_sequences=True))(shared_layer)
      y = TimeDistributed(Dense(self.num_labels, activation=None))(lstm)
      crf = ChainCRF()
      y = crf(y)
    
    self.model = Model(inputs=inputs, outputs=y)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    self.model.compile(loss=crf.loss, optimizer=sgd)

  def get_inputs(self, dataset):
    inputs = [dataset.X1]
    if self.use_gazetteer:
      inputs.append(dataset.X2)
    if self.use_features:
      inputs.append(dataset.X3)
    if self.model_type == 'lstm-crf-cnn' or self.model_type == 'lstm-crf-lstm':
      inputs.append(dataset.X4)
    return inputs

  def fit(self, epochs=20):
    inputs = self.get_inputs(self.dev_dataset)
    self.model.fit(
      x=inputs, y=self.dev_dataset.Y, epochs=epochs, 
      callbacks=[Metrics(self)],
    )

  def flatten(self, y):
    arr = y[:,:,].argmax(axis=-1).flatten()
    arr[arr > 1] = 1
    return arr

  def validate(self):
    inputs = self.get_inputs(self.validate_dataset)
    val_predict = self.model.predict(inputs)
    val_targ = self.validate_dataset.Y
    return val_predict, val_targ, self.validate_dataset.T

  def test(self):
    self.load()
    inputs = self.get_inputs(self.test_dataset)
    val_predict = self.flatten(self.model.predict(inputs))
    val_targ = self.flatten(self.test_dataset.Y)
    return val_targ, val_predict

  def save(self):
    self.model.save('models/' + self.name + '.dat', True)
  
  def load(self):
    from ChainCRF import create_custom_objects
    self.model = keras.models.load_model('models/' + self.name + '.dat', custom_objects=create_custom_objects())
