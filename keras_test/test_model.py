from util import Dataset, WordEmbeddings
from lstm_crf import LstmCrf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from metric import evaluate, get_names, one_hot_to_labels
import np

we       = WordEmbeddings('glove-50')
# test     = Dataset(we, 'dataset/test.txt', max_sentence_len=50, max_token_len=20)
test     = Dataset('dataset/validate.txt', we, max_sentence_len=50, max_token_len=20)

lstm_crf = LstmCrf(
  'lstm-crf-cnn',
  model_type='lstm-crf-cnn',
  dev_dataset=test,
  validate_dataset=test,
  test_dataset=test
)
