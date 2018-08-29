import np
import re
from keras.utils import np_utils

class WordEmbeddings():
  def __init__(self, emb_type):
    self.words  = []
    self.word2Idx  = {}
    self.matrix = None
    self.load_embeddings(emb_type)

  def load_glove(self, dimensions):
    self.matrix = np.zeros((400002, dimensions), dtype=float)
    with open('embeddings/glove.6B.' + str(dimensions) + 'd.txt') as f:
      counter = 2
      for vector in f:
        features = vector.strip().split(' ')
        w = features[0]
        self.words.append(w)
        self.word2Idx[w] = counter
        self.matrix[counter,:] = np.array(features[1:], dtype=float)
        counter += 1

  def load_embeddings(self, emb_type):
    if emb_type == 'glove-50':
      self.load_glove(50)
    elif emb_type == 'glove-100':
      self.load_glove(100)
    elif emb_type == 'glove-200':
      self.load_glove(200)
    elif emb_type == 'glove-300':
      self.load_glove(300)
    elif emb_type == 'word2vec-300':
      self.load_word2vec()

def load_raw_dataset(f, max_sentence_len=50):
  with open(f, 'r') as f:
    sentences = f.read().strip().split('\n\n')
    sentences = [s for s in sentences if not s.startswith('-DOCSTART-')]
    X = [[t.split(' ') for t in s.split('\n') if len(s) > 0] for s in sentences]
    Y = np.zeros((len(X), max_sentence_len, 3))
    T = np.ndarray(shape=(len(X), max_sentence_len), dtype=object)
    for i, s in enumerate(X):
      tkns, labels = [], []
      if len(X[i]) > max_sentence_len:
        X[i] = X[i][:max_sentence_len]

      for j, t in enumerate(s[:max_sentence_len]):
        l = ['O', 'B-PER', 'I-PER'].index(t[1])
        labels.append(np_utils.to_categorical(l, 3))
        tkns.append(t[0])
        X[i][j] = [X[i][j][0]] + X[i][j][2:]
      Y[i,:len(labels)] = np.expand_dims(labels, axis=0)
      T[i,:len(tkns)] = tkns
    return X, Y, T

class Dataset():
  def __init__(self, f, word_embeddings, max_sentence_len=50, max_token_len=20):
    self.we = word_embeddings
    self.num_sentences    = 0
    self.max_sentence_len = max_sentence_len
    self.max_token_len    = max_token_len
    self.X1 = None
    self.X2 = None
    self.X3 = None
    self.Y1 = None
    self.T = None
    self.load_dataset(f)

  def load_dataset(self, f):
    sentences, self.Y1, self.T = load_raw_dataset(f, max_sentence_len=self.max_sentence_len)

    self.num_sentences = len(sentences)
    self.num_features = 7
  
    self.X1 = np.zeros((self.num_sentences, self.max_sentence_len, 1                 ))
    self.X2 = np.zeros((self.num_sentences, self.max_sentence_len, self.num_features ))
    self.X3 = np.zeros((self.num_sentences, self.max_sentence_len, self.max_token_len))

    for i, s in enumerate(sentences):
      chars, tkn_ids, features = [], [], []
      for t in s[:self.max_sentence_len]:
        w = 1
        token = t[0].lower()
        if token in self.we.word2Idx:
          w = self.we.word2Idx[token]

        c = [ord(c) if ord(c) < 128 else 0 for c in list(t[0])]
        if len(t[0]) > self.max_token_len:
          chars.append(c[:self.max_token_len])
        else:
          chars.append(np.pad(c, (0, self.max_token_len-len(t[0])), 'constant'))

        tkn_ids.append([w])
        features.append([int(tkn) for tkn in t[2:9]])
        
      self.X1[i,:len(tkns)] = np.expand_dims(tkn_ids, axis=0)
      self.X2[i,:len(tkns)] = np.array(features, dtype=int)
      self.X3[i,:len(tkns)] = chars
