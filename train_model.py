import sys
import time
import np
from optparse import OptionParser
from keras.callbacks import Callback
from models.util import load_raw_dataset, Dataset, WordEmbeddings, evaluate
from models.lstm_crf import LstmCrf
from models.gazetteer_matcher import gazetteer_predict
from models.hmm import HiddenMarkov

class Metrics(Callback):
  def __init__(self, the_model):
    self.max_f_measure = 0
    self.the_model = the_model

  def on_epoch_end(self, epoch, logs={}):
    val_predict, val_targ, T = self.the_model.validate()
    a, p, r, f1, c, i, m = evaluate(val_predict, val_targ, T)
    
    improved = False
    if f1 > self.max_f_measure:
      self.max_f_measure = f1
      self.the_model.save()
      improved = True

    print('Epoch: %d, Loss: %f, F1: %f %s' % (epoch, logs['loss'], f1, '- IMPROVED' if improved else ''))
 
start = time.time()

parser = OptionParser()
parser.add_option("-f", "--features", dest="use_features", action="store_true")
parser.add_option("-s", "--self-train", dest="self_train", action="store_true")
parser.add_option("-v", "--verbose", dest="verbose", action="store_true")
parser.add_option("-d", "--dataset", dest="dataset")
verbose = False

if len(sys.argv) < 2:
  print('Wrong number of arguments')

model = sys.argv[1]

def print_results(dataset, val_predict, val_targ, T):
  a, p, r, f1, c, i, m = evaluate(val_predict, val_targ, T, verbose=verbose)

  print('----------------')
  print(dataset)
  print('----------------')
  print('Accuracy:', a)
  print('Precision:', p)
  print('Recall:', r)
  print('F1:', f1)
  print('Correct:', c)
  print('Incorrect:', i)
  print('Missed:', m)
  print('')

if not model in [
  'partial_match',
  'exact_match',
  'nb',
  'hmm-1',
  'hmm-2',
  'hmm-3',
  'hmm-4',
  'maxent',
  'crf',
  'lstm-crf',
  'lstm-crf-cnn',
  'lstm-crf-lstm',
]:
  print('Wrong model')
  quit()

(options, args) = parser.parse_args()
options = vars(options)

use_features = not options['use_features'] is None
self_train = not options['self_train'] is None
verbose = not options['verbose'] is None
dataset = options['dataset']

print('================================')
print('')
print('Model:', model)
print('Use features:', use_features)
print('Self train:', self_train)
print('----------------')
print('')

directory = 'dataset'
# directory = 'dataset_good'
# directory = 'dataset'
dev_dataset = directory + '/dev.txt'
validate_dataset = directory + '/validate.txt'
test_dataset = directory + '/test.txt'

if dataset == 'conll':
  dataset = False
  dev_dataset ='conll_dataset/conll_2003.dev.txt'
  validate_dataset ='conll_dataset/conll_2003.validate.txt'
  # test_dataset = 'conll_dataset/conll_2003.test.txt'

if model in ['partial_match', 'exact_match']:
  we = WordEmbeddings('glove-50')
  dev = Dataset(dev_dataset, we, max_sentence_len=50, max_token_len=20)
  validate = Dataset(validate_dataset, we, max_sentence_len=50, max_token_len=20)
  test = Dataset(test_dataset, we, max_sentence_len=50, max_token_len=20)

  mode = 'exact'
  if model == 'partial_match':
    mode = 'partial'

  val_predict = gazetteer_predict(dev.X2, dev.T, mode=mode)
  val_targ = dev.Y
  print_results('Dev', val_predict, val_targ, dev.T)

  val_predict = gazetteer_predict(validate.X2, validate.T, mode=mode)
  val_targ = validate.Y
  print_results('Validate', val_predict, val_targ, validate.T)

  val_predict = gazetteer_predict(test.X2, test.T, mode=mode)
  val_targ = test.Y
  print_results('Test', val_predict, val_targ, test.T)

elif model in ['nb', 'hmm-1', 'hmm-2', 'hmm-3', 'hmm-4']:
  if dataset:
    X, Y, T = load_raw_dataset(dataset)
  else:
    X, Y, T = load_raw_dataset(dev_dataset)

  X2, Y2, T2 = load_raw_dataset(validate_dataset)
  X3, Y3, T3 = load_raw_dataset(test_dataset)

  naive_bayes = False 
  timesteps = 1
  if model == 'hmm-2':
    timesteps = 2
  elif model == 'hmm-3':
    timesteps = 3
  elif model == 'hmm-4':
    timesteps = 4
  elif model == 'nb':
    naive_bayes = True
 
  hmm = HiddenMarkov(
    timesteps, 
    naive_bayes= naive_bayes,
    use_gazetteer=use_features, 
    use_features=use_features, 
    self_train=self_train
  )
  hmm.fit(X, Y)
  
  print_results('Validate', hmm.predict(X2), Y2, T2)
  print_results('Test', hmm.predict(X3), Y3, T3)

elif model in ['maxent', 'crf', 'lstm-crf', 'lstm-crf-cnn', 'lstm-crf-lstm']:
  we = WordEmbeddings('glove-300')

  if dataset:
    dev = Dataset(dataset, we, max_sentence_len=50, max_token_len=20)
  else:
    dev = Dataset(dev_dataset, we, max_sentence_len=50, max_token_len=20)

  validate = Dataset(validate_dataset, we, max_sentence_len=50, max_token_len=20)
  test = Dataset(test_dataset, we, max_sentence_len=50, max_token_len=20)
  
  lstm_crf = LstmCrf(
    model,
    model_type=model, 
    dev_dataset=dev,
    validate_dataset=validate,
    test_dataset=test,
    use_gazetteer=use_features,
    use_features=use_features
  )
  
  lstm_crf.create(we.matrix)
  lstm_crf.fit(callbacks=[Metrics(lstm_crf)], epochs=50)

  lstm_crf.load_best_model()
  val_predict, val_targ, T = lstm_crf.validate()
  print_results('Validate', val_predict, val_targ, T)
  val_predict, val_targ, T = lstm_crf.test()
  print_results('Test', val_predict, val_targ, T)

end = time.time()
print('Time elapsed: ' + str(end - start) + ' seconds.')
print('')

print('================================')
