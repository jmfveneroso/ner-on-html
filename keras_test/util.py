import np
import re
from keras.utils import np_utils
from gensim.models import KeyedVectors

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

  def load_word2vec(dimensions):
    # model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    pass
    
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

class Dataset():
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
    self.Y = None
    self.T = None
    self.load_dataset(f)

  def load_dataset(self, f):
    with open(f, 'r') as f:
      sentences = f.read().strip().split('\n\n')
      sentences = [s for s in sentences if not s.startswith('-DOCSTART-')]
      sentences = [[t.split(' ') for t in s.split('\n') if len(s) > 0] for s in sentences]

      self.labels = ['O', 'B-NAME', 'I-NAME']
      for idx, w in enumerate(self.labels):  
        self.label2Idx[w] = idx
  
      self.num_sentences    = len(sentences)
      self.num_labels       = len(self.labels)

      # self.max_token_len    = max([len(t[0]) for s in sentences for t in s])
      # self.max_sentence_len = max([len(s) for s in sentences])
  
      X1 = np.zeros((self.num_sentences, self.max_sentence_len, 1                 ))
      X2 = np.zeros((self.num_sentences, self.max_sentence_len, 2                 ))
      X3 = np.zeros((self.num_sentences, self.max_sentence_len, self.max_token_len))
      Y1 = np.zeros((self.num_sentences, self.max_sentence_len, self.num_labels   ))
      T  = np.ndarray(shape=(self.num_sentences, self.max_sentence_len), dtype=object)
  
      for i, s in enumerate(sentences):
        chars, tkn_ids, tkns, features, labels  = [], [], [], [], []
        for t in s[:self.max_sentence_len]:
          w = 1
          token = t[0].lower()
          if token in self.we.word2Idx:
            w = self.we.word2Idx[token]

          l = self.label2Idx[t[1]]
          c = [ord(c) if ord(c) < 128 else 0 for c in list(t[0])]
          if len(t[0]) > self.max_token_len:
            chars.append(c[:self.max_token_len])
          else:
            chars.append(np.pad(c, (0, self.max_token_len-len(t[0])), 'constant'))
          tkns.append([t[0]])
          tkn_ids.append([w])
          features.append([int(t[2]), int(t[3])])
          labels.append(np_utils.to_categorical(l, self.num_labels))
	  
        X1[i,:len(tkns)] = np.expand_dims(tkn_ids, axis=0)
        X2[i,:len(tkns)] = np.array(features, dtype=int)
        X3[i,:len(tkns)] = chars
        Y1[i,:len(tkns)] = np.expand_dims(labels, axis=0)
        T [i,:len(tkns)] = tkns
 
      self.X1 = X1
      self.X2 = X2
      self.X3 = X3
      self.Y = Y1
      self.T = T
      return X1, X2, X3, Y1, T

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
    self.Y = None
    self.T = None
    self.load_dataset(f)

  def load_dataset(self, f):
    with open(f, 'r') as f:
      sentences = f.read().strip().split('\n\n')
      sentences = [s for s in sentences if not s.startswith('-DOCSTART-')]
      sentences = [[t.split(' ') for t in s.split('\n') if len(s) > 0] for s in sentences]

      self.labels = ['O', 'I-LOC', 'B-LOC', 'I-PER', 'B-PER', 'I-MISC', 'B-MISC', 'I-ORG', 'B-ORG']
      for idx, w in enumerate(self.labels):  
        self.label2Idx[w] = idx

      f1 = ['EX', 'VBZ', 'VBG', '(', 'RBR', '$', 'UH', "''", 'SYM', ',', 'VB', 'FW', 'JJS', 'WDT', 'JJ', 'MD', '"', 'RBS', 'RP', 'IN', 'CD', 'NNS', 'NN', 'VBN', 'POS', 'WRB', 'WP$', ':', 'JJR', 'NNPS', 'NNP', 'NN|SYM', 'PRP', 'LS', 'RB', 'DT', 'VBP', 'PRP$', 'WP', 'VBD', '.', 'PDT', 'TO', 'CC', ')']
      f2 = ['I-VP', 'I-LST', 'B-SBAR', 'B-ADVP', 'I-NP', 'I-SBAR', 'B-VP', 'O', 'I-CONJP', 'I-ADVP', 'I-PRT', 'I-INTJ', 'I-ADJP', 'B-NP', 'I-PP', 'B-PP', 'B-ADJP']

      # f1 = list(set([t[1] for s in sentences for t in s]))
      # f2 = list(set([t[2] for s in sentences for t in s]))

      self.num_sentences    = len(sentences)
      self.num_labels       = len(self.labels)

      # self.max_token_len    = max([len(t[0]) for s in sentences for t in s])
      # self.max_sentence_len = max([len(s) for s in sentences])
  
      X1 = np.zeros((self.num_sentences, self.max_sentence_len, 1                 ))
      X2 = np.zeros((self.num_sentences, self.max_sentence_len, len(f1) + len(f2) ))
      X3 = np.zeros((self.num_sentences, self.max_sentence_len, self.max_token_len))
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
        X3[i,:len(tkns)] = chars
        Y1[i,:len(tkns)] = np.expand_dims(labels, axis=0)
        T [i,:len(tkns)] = tkns
 
      self.X1 = X1
      self.X2 = X2
      self.X3 = X3
      self.Y = Y1
      self.T = T
      return X1, X2, X3, Y1, T
