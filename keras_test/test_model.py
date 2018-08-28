from util import Dataset, WordEmbeddings
from lstm_crf import LstmCrf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import np

we       = WordEmbeddings('glove-50')
test     = Dataset(we, 'dataset/test.txt', max_sentence_len=50, max_token_len=20)
# test     = Dataset(we, 'dataset/validate.txt', max_sentence_len=50, max_token_len=20)

def flaber(y):
  arr = y[:,:,].argmax(axis=-1).flatten()
  arr[arr > 1] = 1
  return arr

def gazetteer_predict(X, mode='partial'):
  Y = np.zeros((X.shape[0], X.shape[1], 3))

  last_was_name = False
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      if X[i,j,1] == 0:
        last_was_name = False 

      # Exact.
      if mode == 'exact':
        if X[i,j,0] == 1:
          if last_was_name:
            Y[i,j] = np.array([0, 0, 1])
          else:
            Y[i,j] = np.array([0, 1, 0])
            last_was_name = True

      # Partial .
      else:
        if X[i,j,1] == 1:
          if last_was_name:
            Y[i,j] = np.array([0, 0, 1])
          else:
            Y[i,j] = np.array([0, 1, 0])
            last_was_name = True
  return Y      

# val_predict = flaber(gazetteer_predict(test.X2))
# val_targ    = flaber(test.Y)
# 
# _val_f1 = f1_score(val_targ, val_predict)
# _val_recall = recall_score(val_targ, val_predict)
# _val_precision = precision_score(val_targ, val_predict)
# print(" — f1: %f — precision: %f — recall %f" % (_val_f1, _val_precision, _val_recall))
# quit()

lstm_crf = LstmCrf(
  'lstm-crf-cnn',
  model_type='lstm-crf-cnn',
  dev_dataset=test,
  validate_dataset=test,
  test_dataset=test
)
lstm_crf.print_names()
