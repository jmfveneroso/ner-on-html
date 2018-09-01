import sys
from optparse import OptionParser
from metric import evaluate, get_names, one_hot_to_labels
from util import load_raw_dataset
from util import Dataset, WordEmbeddings
from lstm_crf import LstmCrf
from gazetteer_matcher import gazetteer_predict
from hmm_proto import HiddenMarkov

parser = OptionParser()
parser.add_option("-g", "--gazetteer", dest="use_gazetteer", action="store_true")
parser.add_option("-f", "--features", dest="use_features", action="store_true")
parser.add_option("-s", "--self-train", dest="self_train", action="store_true")
parser.add_option("-r", "--reduced-dataset", dest="reduced_dataset", action="store_true")

if len(sys.argv) < 2:
  print('Wrong number of arguments')

model = sys.argv[1]

if not model in [
  'partial_match',
  'exact_match',
  'hmm-1',
  'hmm-2',
  'hmm-3',
  'hmm-4',
  'crf',
  'lstm-crf',
  'lstm-crf-cnn',
  'lstm-crf-lstm',
]:
  print('Wrong model')
  quit()

(options, args) = parser.parse_args()
options = vars(options)

use_gazetteer = not options['use_gazetteer'] is None
use_features = not options['use_features'] is None
self_train = not options['self_train'] is None

if model in ['partial_match', 'exact_match']:
  we = WordEmbeddings('glove-50')
  test = Dataset('dataset/test.txt', we, max_sentence_len=50, max_token_len=20)

  mode = 'exact'
  if model == 'partial_match':
    mode = 'partial'

  val_predict = gazetteer_predict(test.X2, test.T, mode=mode)
  val_targ = test.Y
  a, p, r, f1, c, i, m = evaluate(val_predict, val_targ, test.T)

  print('Accuracy:', a)
  print('Precision:', p)
  print('Recall:', r)
  print('F1:', f1)
  print('Correct:', c)
  print('Incorrect:', i)
  print('Missed:', m)

elif model in ['hmm-1', 'hmm-2', 'hmm-3', 'hmm-4']:
  X, Y, T    = load_raw_dataset('dataset/dev.txt')
  X2, Y2, T2 = load_raw_dataset('dataset/validate.txt')
 
  timesteps = 1
  if model == 'hmm-2':
    timesteps = 2
  elif model == 'hmm-3':
    timesteps = 3
  elif model == 'hmm-4':
    timesteps = 4
 
  hmm = HiddenMarkov(
    timesteps, 
    use_gazetteer=use_gazetteer, 
    use_features=use_features, 
    self_train=self_train
  )
  hmm.fit(X, Y)
  
  a, p, r, f1, c, i, m = evaluate(hmm.predict(X2), Y2, T2)
  
  print('Accuracy:', a)
  print('Precision:', p)
  print('Recall:', r)
  print('F1:', f1)
  print('Correct:', c)
  print('Incorrect:', i)
  print('Missed:', m)

elif model in ['crf', 'lstm-crf', 'lstm-crf-cnn', 'lstm-crf-lstm']:
  we = WordEmbeddings('glove-50')

  dev = None
  if not options['reduced_dataset'] is None:
    dev = Dataset('dataset/small_dev.txt', we, max_sentence_len=50, max_token_len=20)
  else:
    dev = Dataset('dataset/dev.txt', we, max_sentence_len=50, max_token_len=20)

  validate = Dataset('dataset/validate.txt', we, max_sentence_len=50, max_token_len=20)
  test     = Dataset('dataset/test.txt',     we, max_sentence_len=50, max_token_len=20)
  
  lstm_crf = LstmCrf(
    model,
    model_type=model, 
    dev_dataset=dev,
    validate_dataset=validate,
    test_dataset=test,
    use_gazetteer=use_gazetteer,
    use_features=use_gazetteer
  )
  
  lstm_crf.create(we.matrix)
  lstm_crf.fit(epochs=10)
