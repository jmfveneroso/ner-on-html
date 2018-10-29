import keras
import np
import re
from models.util import Dataset, WordEmbeddings, evaluate, get_names, one_hot_to_labels
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, TimeDistributed, Bidirectional, LSTM, Embedding, concatenate, Conv1D, GlobalMaxPooling1D
from models.ChainCRF import ChainCRF, create_custom_objects
from keras.utils import np_utils
from keras import optimizers
  
class LstmCrf():
  def __init__(
    self, 
    name,
    lstm_cells=100, 
    use_gazetteer=True, 
    use_features=True, 
    char_embedding_size=30,
    num_cnn_filters=30,
    char_lstm_size=25,
    model_type=['lstm-crf', 'lstm-crf-cnn', 'lstm-crf-lstm', 'crf', 'maxent'][0],
    dev_dataset=None,
    validate_dataset=None,
    test_dataset=None,
  ):
    self.name = name
    self.lstm_cells = lstm_cells
    self.num_cnn_filters = num_cnn_filters
    self.char_lstm_size = char_lstm_size
    self.model_type = model_type
    self.dev_dataset = dev_dataset
    self.validate_dataset = validate_dataset
    self.test_dataset = test_dataset
    self.use_gazetteer = use_gazetteer
    self.use_features = use_features
    self.num_labels = self.dev_dataset.Y.shape[2]

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
    if self.model_type == 'lstm-crf-cnn' or self.model_type == 'lstm-crf-lstm':
      x = Input(shape=(x4_shape[1], x4_shape[2]), dtype='int32')
      inputs.append(x)

      char_emb = TimeDistributed(Embedding(
        input_dim=char_embedding_matrix.shape[0],  
        output_dim=char_embedding_matrix.shape[1],
        weights=[char_embedding_matrix], 
        trainable=True
      ))(x)
      
      # LSTM char embeddings from Lample et al., 2016.
      if self.model_type == 'lstm-crf-lstm':
        char_emb = TimeDistributed(Bidirectional(LSTM(self.char_lstm_size, return_sequences=False)))(char_emb)

      # CNN char embeddings from Ma and Hovy, 2016.
      elif self.model_type == 'lstm-crf-cnn':
        char_emb = TimeDistributed(Conv1D(self.num_cnn_filters, 3))(char_emb)
        char_emb = TimeDistributed(GlobalMaxPooling1D())(char_emb)
      
      shared_layer.append(char_emb)
   
    if len(shared_layer) > 1:
      shared_layer = concatenate(shared_layer)
    else:
      shared_layer = shared_layer[0]

    if self.model_type == 'crf':
      y = TimeDistributed(Dense(self.num_labels, activation=None))(shared_layer)
      crf = ChainCRF()
      y = crf(y)
    elif self.model_type == 'maxent':
      y = TimeDistributed(Dense(self.num_labels, activation='softmax'))(shared_layer)
    else:
      lstm = Bidirectional(LSTM(self.lstm_cells, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(shared_layer)
      y = TimeDistributed(Dense(self.num_labels, activation=None))(lstm)
      crf = ChainCRF()
      y = crf(y)

    self.model = Model(inputs=inputs, outputs=y)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    if self.model_type == 'maxent':
      self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
    else:
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

  def fit(self, callbacks=[], epochs=20):
    inputs = self.get_inputs(self.dev_dataset)

    history = self.model.fit(
      x=inputs, y=self.dev_dataset.Y, epochs=epochs, 
      callbacks=callbacks, verbose=0,
      # batch_size=10, 
    )

  def validate(self):
    inputs = self.get_inputs(self.validate_dataset)
    val_predict = self.model.predict(inputs)
    val_targ = self.validate_dataset.Y
    return val_predict, val_targ, self.validate_dataset.T

  def test(self):
    inputs = self.get_inputs(self.test_dataset)
    val_predict = self.model.predict(inputs)
    val_targ = self.test_dataset.Y
    return val_predict, val_targ, self.test_dataset.T

  def save(self):
    self.model.save('model_data/' + self.name + '.dat', True)
  
  def load_best_model(self):
    self.model = keras.models.load_model('model_data/' + self.name + '.dat', custom_objects=create_custom_objects())
