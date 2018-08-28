import keras
import np
import re
from matplotlib import pyplot as plt
from util import Dataset, WordEmbeddings

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, SimpleRNN, TimeDistributed, GRU, Bidirectional, LSTM, Embedding, concatenate, Conv1D, GlobalMaxPooling1D
from ChainCRF import ChainCRF
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import Callback
from keras import optimizers
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
  def __init__(self, the_model):
    self.max_f_measure = 0
    self.the_model = the_model

  def on_train_begin(self, logs={}):
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []
   
  def on_epoch_end(self, epoch, logs={}):
    val_targ, val_predict = self.the_model.validate()

    # labels = self.the_model.validate_dataset.labels[idx]
    # precision =
    # recall = 
    # f1 = 
    # for 

    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    print(" — f1: %f — precision: %f — recall %f" % (_val_f1, _val_precision, _val_recall))

    if _val_f1 > self.max_f_measure:
      self.max_f_measure = _val_f1
      self.the_model.save()
    else:
      print("No improvement")
   
class LstmCrf():
  def __init__(
    self, 
    name,
    lstm_cells=100, 
    char_embedding_size=30,
    num_cnn_filters=30,
    num_labels=3,
    char_lstm_size=25,
    model_type=['lstm-crf', 'lstm-crf-cnn', 'lstm-crf-lstm'][0],
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

  def create(self, word_embedding_matrix):
    x1_shape = self.dev_dataset.X1.shape
    x2_shape = self.dev_dataset.X2.shape
    x3_shape = self.dev_dataset.X3.shape

    char_embedding_matrix = np.random.randn(128, 30)

    x = Input(shape=(x1_shape[1], x1_shape[2]), dtype='int32')
    x2 = Input(shape=(x2_shape[1], x2_shape[2]), dtype='float32')
    inputs = [x, x2]
    
    # Word embeddings.
    word_emb = Embedding(
      input_dim=word_embedding_matrix.shape[0], 
      output_dim=word_embedding_matrix.shape[1], 
      weights=[word_embedding_matrix], 
      trainable=False
    )(Flatten()(x))

    # Feature embeddings.
    # feature_emb = Embedding(
    #   input_dim=x2_shape[2], output_dim=x2_shape[2]
    # )(x2)
    feature_emb = x2

    shared_layer = [word_emb, feature_emb]

    # Char embeddings.
    if self.model_type != 'lstm-crf' and self.model_type != 'crf':
      x = Input(shape=(x3_shape[1], x3_shape[2]), dtype='int32')
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
    # adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    self.model.compile(loss=crf.loss, optimizer=sgd)

  def fit(self, epochs=20):
    inputs = [self.dev_dataset.X1, self.dev_dataset.X2, self.dev_dataset.X3]
    if self.model_type == 'lstm-crf' or self.model_type == 'crf':
      inputs = [self.dev_dataset.X1, self.dev_dataset.X2]

    self.model.fit(
	      x=inputs, y=self.dev_dataset.Y, epochs=epochs, 
      callbacks=[Metrics(self)],
    )

  def flatten(self, y):
    arr = y[:,:,].argmax(axis=-1).flatten()
    arr[arr > 1] = 1
    return arr

  def validate(self):
    inputs = [self.validate_dataset.X1, self.validate_dataset.X2, self.validate_dataset.X3]
    if self.model_type == 'lstm-crf' or self.model_type == 'crf':
      inputs = [self.validate_dataset.X1, self.validate_dataset.X2]

    val_predict = self.flatten(self.model.predict(inputs))
    val_targ = self.flatten(self.validate_dataset.Y)
    return val_targ, val_predict

  def test(self):
    self.load()
    inputs = [self.test_dataset.X1, self.test_dataset.X2, self.test_dataset.X3]
    if self.model_type == 'lstm-crf' or self.model_type == 'crf':
      inputs = [self.test_dataset.X1, self.test_dataset.X2]

    val_predict = self.flatten(self.model.predict(inputs))
    val_targ = self.flatten(self.test_dataset.Y)
    return val_targ, val_predict

  def remove_titles(self, text):
    text = re.sub('(M\. Sc\.)|(M\.Sc\.)|(MS\. SC\.)|(MS\.SC\.)', '', text)
    text = re.sub('(M\. Ed\.)|(M\.Ed\.)|(M\. ED\.)|(M\.ED\.)', '', text)
    text = re.sub('(sc\. nat\.)|(sc\.nat\.)', '', text)
    text = re.sub('(rer\. nat\.)|(rer\.nat\.)|(rer nat)|(rer\. nat)', '', text)
    text = re.sub('(i\. R\.)', '', text)
    text = re.sub('(PD )|( PD )', '', text)
    text = re.sub('(Sc\. Nat\.)|(Sc\.Nat\.)|(SC\. NAT\.)|(SC\.NAT\.)', '', text)
    text = re.sub('(Sc\. Nat)|(Sc\.Nat)|(SC\. NAT)|(SC\.NAT)', '', text)
    text = re.sub('(MD\.)|(Md\.)|(Md )', '', text)
    text = re.sub('(B\. Sc\.)|(B\.Sc\.)|(BS\. SC\.)|(BS\.SC\.)', '', text)
    text = re.sub('(B\. Sc)|(B\.Sc)|(BS\. SC)|(BS\.SC)', '', text)
    text = re.sub('(Ph\. D\.)|(Ph\.D\.)|(PH\. D\.)|(PH\.D\.)', '', text)
    text = re.sub('(Ph\. D)|(Ph\.D)|(PH\. D)|(PH\.D)', '', text)
    text = re.sub('(Ed\. D\.)|(Ed\.D\.)|(ED\. D\.)|(ED\.D\.)', '', text)
    text = re.sub('(Ed\. D)|(Ed\.D)|(ED\. D)|(ED\.D)', '', text)
    text = re.sub('(M\. S\.)|(M\.S\.)', '', text)
    text = re.sub('(Hon\.)', '', text)
    text = re.sub('(a\.D\.)', '', text)
    text = re.sub('(em\.)', '', text)
    text = re.sub('(apl\.)|(Apl\.)', '', text)
    text = re.sub('(apl\.)|(Apl\.)', '', text)
    text = re.sub('(Prof\.dr\.)', '', text)
    text = re.sub('(Conf\.dr\.)', '', text)
    text = re.sub('(Asist\.dr\.)', '', text)
    return text.strip()

  def print_names(self):
    self.load()
    inputs = [self.test_dataset.X1, self.test_dataset.X2, self.test_dataset.X3]
    if self.model_type == 'lstm-crf' or self.model_type == 'crf':
      inputs = [self.test_dataset.X1, self.test_dataset.X2]

    tokens = self.test_dataset.T.flatten()
    val_predict = self.flatten(self.model.predict(inputs))
    val_targ = self.flatten(self.test_dataset.Y)

    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    print(" — f1: %f — precision: %f — recall %f" % (_val_f1, _val_precision, _val_recall))
        
    val_predict = self.model.predict(inputs).argmax(axis=-1).flatten()

    names = []
    name = []
    for i, t in enumerate(tokens):
      if val_predict[i] == 1:
        if len(name) > 0:
          names.append(' '.join(name))
          name = []
        if not t is None:
          name.append(t[0])
      elif val_predict[i] == 2:
        if not t is None:
          name.append(t[0])
      elif len(name) > 0:
        names.append(' '.join(name))
        name = []
    if len(name) > 0:
      names.append(' '.join(name))

    val_predict = self.test_dataset.Y.argmax(axis=-1).flatten()
    names2 = []
    name = []
    for i, t in enumerate(tokens):
      if val_predict[i] == 1:
        if len(name) > 0:
          names2.append(' '.join(name))
          name = []
        if not t is None:
          name.append(t[0])
      elif val_predict[i] == 2:
        if not t is None:
          name.append(t[0])
      elif len(name) > 0:
        names2.append(' '.join(name))
        name = []
    if len(name) > 0:
      names2.append(' '.join(name))

    names = list(set([self.remove_titles(n) for n in names]))
    names2 = list(set(names2))

    correct = []
    incorrect = []
    for n in names:    
      if n in names2:
        correct.append(n)
      else:
        incorrect.append(n)

    missed = []
    for n in names2:    
      if not n in names:
        missed.append(n)

    print('precision:', len(correct) / float(len(names)))
    print('recall:',    len(correct) / float(len(names2)))
    print('correct:', len(correct))
    print('incorrect:', len(incorrect))
    print('missed:', len(missed))

    for n in missed:
      print(n)

    print('\n\n\n\n')

    for n in incorrect:
      print(n)

  def save(self):
    self.model.save('models/' + self.name + '.dat', True)
  
  def load(self):
    from ChainCRF import create_custom_objects
    self.model = keras.models.load_model('models/' + self.name + '.dat', custom_objects=create_custom_objects())
