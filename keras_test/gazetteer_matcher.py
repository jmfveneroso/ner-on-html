from util import Dataset, WordEmbeddings
from metric import evaluate, get_names, one_hot_to_labels
from tokenizer import is_punctuation
import np

we       = WordEmbeddings('glove-50')
test     = Dataset('dataset/validate.txt', we, max_sentence_len=50, max_token_len=20)
# test     = Dataset(we, 'dataset/test.txt', max_sentence_len=50, max_token_len=20)

def gazetteer_predict(X, T, mode='partial'):
  Y = np.zeros((X.shape[0], X.shape[1], 4))

  last_was_name = False
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      if X[i,j,1] == 0:
        last_was_name = False 

      if not T[i,j] is None and is_punctuation(T[i,j]):
        Y[i,j] = np.array([0, 0, 0, 1])

      # Exact.
      elif mode == 'exact':
        if X[i,j,0] == 1:
          if last_was_name:
            Y[i,j] = np.array([0, 0, 1, 0])
          else:
            Y[i,j] = np.array([0, 1, 0, 0])
            last_was_name = True

      # Partial .
      else:
        if X[i,j,1] == 1:
          if last_was_name:
            Y[i,j] = np.array([0, 0, 1, 0])
          else:
            Y[i,j] = np.array([0, 1, 0, 0])
            last_was_name = True
  return Y      
