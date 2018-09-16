#!/usr/bin/python
# coding=UTF-8

import np
from keras.utils import np_utils
from keras.callbacks import Callback
from models.util import WordEmbeddings
from models.lstm_crf import LstmCrf

def is_punctuation(text):
  return re.match("^[\,\;\:\-\"\(\)“”]$", text)

def print_results(dataset, val_predict, val_targ, T, verbose=False):
    correct = 0
    incorrect = 0
    missed = 0
    for i, sentence in enumerate(val_targ):
      for j, tkn in enumerate(val_targ[i]):
        if np.sum(val_targ[i,j]) > 0:
          y_hat = int(np.argmax(val_predict[i,j], axis=-1))
          y = int(np.argmax(val_targ[i,j], axis=-1))

          if y_hat == y and y != 0:
            correct += 1

          elif y_hat != y and y == 0:
            incorrect += 1

          elif y_hat != y and y > 0:
            missed += 1
            if y_hat > 0:
              incorrect += 1

    p  = correct / float(correct + incorrect)
    r  = correct / float(correct + missed)
    f1 = 2 * p * r / float(p + r)

    if verbose:
      print('----------------')
      print(dataset)
      print('----------------')
      print('Correct:', correct)
      print('Incorrect:', incorrect)
      print('Missed:', missed)
      print('Precision:', p)
      print('Recall:', r)
      print('F1:', f1)

    return correct, incorrect, missed, p, r, f1

class Metrics(Callback):
  def __init__(self, the_model):
    self.max_f_measure = 0
    self.the_model = the_model

  def on_epoch_end(self, epoch, logs={}):
    val_predict, val_targ, T = self.the_model.validate()

    correct, incorrect, missed, p, r, f1 = print_results('Validate', val_predict, val_targ, T, verbose=False)

    improved = False
    if f1 > self.max_f_measure:
      self.max_f_measure = f1
      self.the_model.save()
      improved = True

    print('Epoch: %d, Loss: %f, F1: %f %s' % (epoch, logs['loss'], f1, '- IMPROVED' if improved else ''))
 
class Conll2003():
  def __init__(self, word_embeddings, f, max_sentence_len=0, max_token_len=0):
    self.we = word_embeddings
    self.labels = []
    self.label2Idx = {}
    self.num_sentences    = 0
    self.num_labels       = 0
    self.max_sentence_len = max_sentence_len
    self.max_token_len    = max_token_len
    self.X1 = None
    self.X2 = None
    self.X3 = None
    self.X4 = None
    self.Y = None
    self.T = None
    self.load_dataset(f)

  def load_dataset(self, f):
    with open(f, 'r') as f:
      sentences = f.read().strip().split('\n\n')
      sentences = [s for s in sentences if not s.startswith('-DOCSTART-')]
      sentences = [[t.split(' ') for t in s.split('\n') if len(s) > 0] for s in sentences]

      self.labels = [
        'O', 
        'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 
        'B-LOC', 'B-PER', 'B-ORG', 'B-MISC'
      ]
      l = ['O', 'B-PER', 'I-PER', 'PUNCT-O']

      for idx, w in enumerate(self.labels):  
        self.label2Idx[w] = idx

      f1 = [
        'EX', 'VBZ', 'VBG', '(', 'RBR', '$', 'UH', "''", 'SYM', ',', 
        'VB', 'FW', 'JJS', 'WDT', 'JJ', 'MD', '"', 'RBS', 'RP', 'IN', 
        'CD', 'NNS', 'NN', 'VBN', 'POS', 'WRB', 'WP$', ':', 'JJR', 
        'NNPS', 'NNP', 'NN|SYM', 'PRP', 'LS', 'RB', 'DT', 'VBP', 
        'PRP$', 'WP', 'VBD', '.', 'PDT', 'TO', 'CC', ')'
      ]

      f2 = [
        'I-VP', 'I-LST', 'B-SBAR', 'B-ADVP', 'I-NP', 'I-SBAR', 'B-VP', 
        'O', 'I-CONJP', 'I-ADVP', 'I-PRT', 'I-INTJ', 'I-ADJP', 'B-NP', 
        'I-PP', 'B-PP', 'B-ADJP'
      ]

      self.num_sentences    = len(sentences)
      self.num_labels       = len(self.labels)

      X1 = np.zeros((self.num_sentences, self.max_sentence_len, 1                 ))
      X2 = np.zeros((self.num_sentences, self.max_sentence_len, len(f1) + len(f2) ))
      X3 = np.zeros((self.num_sentences, self.max_sentence_len, 1                 ))
      X4 = np.zeros((self.num_sentences, self.max_sentence_len, self.max_token_len))
      Y1 = np.zeros((self.num_sentences, self.max_sentence_len, self.num_labels   ))
      T  = np.ndarray(shape=(self.num_sentences, self.max_sentence_len), dtype=object)
  
      for i, s in enumerate(sentences):
        chars, tkn_ids, tkns, features, labels  = [], [], [], [], []
        for t in s[:self.max_sentence_len]:
          w = 1
          token = t[0].lower()
          if token in self.we.word2Idx:
            w = self.we.word2Idx[token]

          l = self.label2Idx[t[3]]
          c = [ord(c) if ord(c) < 128 else 0 for c in list(t[0])]
          if len(t[0]) > self.max_token_len:
            chars.append(c[:self.max_token_len])
          else:
            chars.append(np.pad(c, (0, self.max_token_len-len(t[0])), 'constant'))
          tkns.append([t[0]])
          tkn_ids.append([w])

          f1_ = np_utils.to_categorical(int(f1.index(t[1])), len(f1))
          f2_ = np_utils.to_categorical(int(f2.index(t[2])), len(f2))
          features.append(np.concatenate((f1_, f2_)))
          labels.append(np_utils.to_categorical(l, self.num_labels))
          
        X1[i,:len(tkns)] = np.expand_dims(tkn_ids, axis=0)
        X2[i,:len(tkns)] = np.array(features, dtype=int)
        X4[i,:len(tkns)] = chars
        Y1[i,:len(tkns)] = np.expand_dims(labels, axis=0)
        T [i,:len(tkns)] = tkns
 
      self.X1 = X1
      self.X2 = X2
      self.X3 = X3
      self.X4 = X4
      self.Y = Y1
      self.T = T

we       = WordEmbeddings('glove-50')
dev      = Conll2003(we, 'conll-2003/eng.train', max_sentence_len=50, max_token_len=20)
validate = Conll2003(we, 'conll-2003/eng.testa', max_sentence_len=50, max_token_len=20)
test     = Conll2003(we, 'conll-2003/eng.testb', max_sentence_len=50, max_token_len=20)

lstm_crf = LstmCrf(
  'lstm-crf',
  model_type='lstm-crf', 
  dev_dataset=dev,
  validate_dataset=validate,
  test_dataset=test,
  use_features=False,
  use_gazetteer=True,
)
  
lstm_crf.create(we.matrix)
lstm_crf.fit([Metrics(lstm_crf)], epochs=10)

lstm_crf.load_best_model()
val_predict, val_targ, T = lstm_crf.validate()
print_results('Validate', val_predict, val_targ, T, verbose=True)
val_predict, val_targ, T = lstm_crf.test()
print_results('Test', val_predict, val_targ, T, verbose=True)
