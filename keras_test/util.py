import np
from keras.utils import np_utils

class Dataset():
  def __init__(self):
    self.words  = []
    self.labels = []
    self.word2Idx  = {}
    self.label2Idx = {}
    self.num_sentences    = 0
    self.num_labels       = 0
    self.max_sentence_len = 0
    self.max_token_len    = 0
    self.embedding_matrix = None
    
  def load_word_embeddings(self, f):
    embedding_size = 30
    self.embedding_matrix = np.random.randn(13, embedding_size)
    return self.embedding_matrix

  def load_dataset(self, f):
    with open(f, 'r') as f:
      sentences = f.read().strip().split('\n\n')
      sentences = [[t.split(' ') for t in s.split('\n')] for s in sentences]
      self.words  = [None] + list(set([t[0] for s in sentences for t in s]))
      self.labels = list(set([t[1] for s in sentences for t in s]))
        
      for idx, w in enumerate(self.words):  
        self.word2Idx[w] = idx
  
      for idx, w in enumerate(self.labels):  
        self.label2Idx[w] = idx
  
      self.num_sentences    = len(sentences)
      self.num_labels       = len(self.labels)
      self.max_sentence_len = max([len(s) for s in sentences])
      self.max_token_len    = max([len(t[0]) for s in sentences for t in s])
  
      X1 = np.zeros((self.num_sentences, self.max_sentence_len, 1                 ))
      X2 = np.zeros((self.num_sentences, self.max_sentence_len, self.max_token_len))
      Y1 = np.zeros((self.num_sentences, self.max_sentence_len, self.num_labels   ))
  
      for i, s in enumerate(sentences):
        chars, tokens, labels  = [], [], []
        for t in s:
          w, l = self.word2Idx[t[0]], self.label2Idx[t[1]]
          c = [ord(c) for c in list(t[0])]
          chars.append(np.pad(c, (0, self.max_token_len-len(t[0])), 'constant'))
          tokens.append([w])
          labels.append(np_utils.to_categorical(l, self.num_labels))
  
        X1[i,:len(s)] = np.expand_dims(tokens, axis=0)
        X2[i,:len(s)] = chars
        Y1[i,:len(s)] = np.expand_dims(labels, axis=0)
  
      return X1, X2, Y1
