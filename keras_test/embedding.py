from gensim.models import KeyedVectors
import np

class WordEmbeddings:
  def __init__(emb_type=''):
    self.words    = []
    self.word2Idx = {}
    self.embedding_matrix = None

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
  
  def load_glove(dimensions):
    self.embedding_matrix = np.zeros((400001, dimensions), dtype=float)
    with open('embeddings/glove.6B.' + str(dimensions) + 'd.txt') as f:
      counter = 1
      for vector in f:
        features = vector.strip().split(' ')
        w = features[0]
        words.append(w)
        word2Idx[w] = counter
        embedding_matrix[counter,:] = np.array(features[1:], dtype=float)
        counter += 1

  def load_word2vec(dimensions):
    # model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    pass
