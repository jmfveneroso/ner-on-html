import np
import re
from keras.utils import np_utils
# np.set_printoptions(suppress=True)

num_labels = 3

def is_punctuation(text):
  return re.match("^[\,\;\:\-\"\(\)“”；]$", text)

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

def load_raw_dataset(f, max_sentence_len=50):
  with open(f, 'r') as f:
    sentences = f.read().strip().split('\n\n')
    sentences = [s for s in sentences if not s.startswith('-DOCSTART-')]
    X = [[t.split(' ') for t in s.split('\n') if len(s) > 0] for s in sentences]
    Y = np.zeros((len(X), max_sentence_len, num_labels))
    T = np.ndarray(shape=(len(X), max_sentence_len), dtype=object)
    for i, s in enumerate(X):
      tkns, labels = [], []
      if len(X[i]) > max_sentence_len-1:
        X[i] = X[i][:max_sentence_len-1]

      for j, t in enumerate(s[:max_sentence_len-1]):
        # l = ['O', 'B-PER', 'I-PER', 'PUNCT'].index(t[1])
        l = ['O', 'B-PER', 'I-PER'].index(t[1])
        labels.append(np_utils.to_categorical(l, num_labels))
        tkns.append(t[0])
        X[i][j] = [X[i][j][0]] + X[i][j][2:]

      # EOS.
      # X[i].append(['EOS'] + ['0'] * 12)
      # labels.append(np_utils.to_categorical(0, num_labels))
      # tkns.append('EOS')

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
    self.X4 = None
    self.Y = None
    self.T = None
    self.load_dataset(f)

  def to_one_hot(self, val, num_classes=15):
    if val >= num_classes:
      val = num_classes - 1
    result = [0] * num_classes
    result[val] = 1
    return result

  def load_dataset(self, f):
    sentences, self.Y, self.T = load_raw_dataset(f, max_sentence_len=self.max_sentence_len)

    self.num_sentences = len(sentences)

    # HTML features are ignored. They are only useful for HMM self training.
    self.num_features = 6
  
    self.X1 = np.zeros((self.num_sentences, self.max_sentence_len, 1                     ))
    self.X2 = np.zeros((self.num_sentences, self.max_sentence_len, 32                    ))
    self.X3 = np.zeros((self.num_sentences, self.max_sentence_len, self.num_features     ))
    self.X4 = np.zeros((self.num_sentences, self.max_sentence_len, self.max_token_len    ))

    for i, s in enumerate(sentences):
      chars, tkn_ids, gazetteer, features = [], [], [], []
      for t in s[:self.max_sentence_len]:
        w = 1
        token = t[1] # This is the lowercase unaccented token.
        if token in self.we.word2Idx:
          w = self.we.word2Idx[token]

        c = [ord(c) if ord(c) < 128 else 0 for c in list(t[0])]
        if len(t[0]) > self.max_token_len:
          chars.append(c[:self.max_token_len])
        else:
          chars.append(np.pad(c, (0, self.max_token_len-len(t[0])), 'constant'))

        tkn_ids.append([w])

        f = [int(tkn) for tkn in t[2:4]]
        f += self.to_one_hot(int(t[4]))
        f += self.to_one_hot(int(t[5]))
        gazetteer.append(f)

        f = [int(tkn) for tkn in t[6:12]]
        features.append(f)
        
      self.X1[i,:len(tkn_ids)] = np.expand_dims(tkn_ids, axis=0)
      self.X2[i,:len(tkn_ids)] = np.array(gazetteer, dtype=int)
      self.X3[i,:len(tkn_ids)] = np.array(features, dtype=int)
      self.X4[i,:len(tkn_ids)] = chars

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

      self.labels = ['O', 'B-PER', 'I-PER', 'PUNCT']
      # self.labels = ['O', 'I-PER', 'PUNCT']

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

          l = 0
          if t[3] in self.label2Idx:
            l = self.label2Idx[t[3]]
          elif is_punctuation(token):
            l = 3

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

def remove_titles(text):
  name = []
  for tkn in text.split(' '):
    titles = ['m\.sc\.','sc\.nat\.','rer\.nat\.','sc\.nat\.','md\.',
      'b\.sc\.', 'bs\.sc\.', 'ph\.d\.', 'ed\.d\.', 'm\.s\.', 'hon\.', 
      'a\.d\.', 'em\.', 'apl\.', 'prof\.', 'prof\.dr\.', 'conf\.dr\.',
      'asist\.dr\.', 'dr\.', 'mr\.', 'mrs\.', 'lect\.dr\.',
      'dr', 'professor', 'mr', 'mrs', 'ing\.'
    ]

    match = False
    for title in titles:
      if re.match('^' + title + '$', tkn, re.IGNORECASE):
        match = True
        break

    if not match:
      name.append(tkn)
        
  return ' '.join(name)

def get_names(val_predict, sentences, x=False):
  names, name = [], []
  for i, tokens in enumerate(sentences):
    added_name = False
    for j, t in enumerate(tokens):
      if val_predict[i][j] == 1:
        if len(name) > 0:
          names.append(' '.join(name))
          added_name = added_name or len(name) == 1
          name = []
        if not t is None:
          name.append(t)
      elif val_predict[i][j] == 2:
        if not t is None:
          name.append(t)
      # if val_predict[i][j] == 1 or val_predict[i][j] == 2:
      #   if not t is None:
      #     name.append(t)
      # elif val_predict[i][j] == 3:
      #   if len(name) > 0 and j > 0 and j+1 < len(tokens):
      #     if (val_predict[i][j-1] == 1 or val_predict[i][j-1] == 2) and val_predict[i][j+1] == 2:
      #       name.append(t)
      #     elif len(name) > 0:
      #       names.append(' '.join(name))
      #       added_name = added_name or len(name) == 1
      #       name = []
      elif len(name) > 0:
        names.append(' '.join(name))
        added_name = added_name or len(name) == 1
        name = []

    if len(name) > 0:
      names.append(' '.join(name))
      added_name = added_name or len(name) == 1

    if added_name and x:
      print(names[-5:])
      print(tokens)
      print(val_predict[i])

  # names = [remove_titles(n) for n in names]
  # return [n for n in names if len(n.split(' ')) > 1]
  return names

def one_hot_to_labels(Y):
  res = np.zeros((Y.shape[0], Y.shape[1]), dtype=int)
  for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
      if np.sum(Y[i,j]) == 0:
        res[i,j] = -1
      else: 
        res[i,j] = int(np.argmax(Y[i,j], axis=-1))
  return res

def evaluate_old(val_predict, val_targ, tokens, verbose=False):
  val_predict = one_hot_to_labels(val_predict)
  val_targ = one_hot_to_labels(val_targ)

  confusion_matrix = np.zeros((num_labels,num_labels))

  accuracy = 0
  total = 0
  for i in range(val_predict.shape[0]):
    for j in range(val_predict.shape[1]):
      if val_targ[i,j] == -1:
        continue

      y_hat = val_predict[i,j]
      y     = val_targ[i,j]
      confusion_matrix[y_hat,y] += 1

      if y_hat == y:
        accuracy += 1 
      total += 1 

  predicted_names = list(set(get_names(val_predict, tokens)))
  target_names = list(set(get_names(val_targ, tokens)))
  
  correct = []
  incorrect = []
  for n in predicted_names:    
    if n in target_names:
      correct.append(n)
    else:
      incorrect.append(n)
      if verbose:
        print(n)
  
  if verbose:
    print('========================')
  missed = []
  for n in target_names:    
    if not n in predicted_names:
      missed.append(n)
      if verbose:
        print(n)

  if verbose:
    print(confusion_matrix)

  accuracy = accuracy / float(total)
  precision = len(correct) / float(len(predicted_names))
  recall = len(correct) / float(len(target_names))
  correct = len(correct)
  incorrect = len(incorrect)
  missed = len(missed)
  f1 = 2 * precision * recall / (precision + recall)

  return accuracy, precision, recall, f1, correct, incorrect, missed 

def get_named_entities(tags):
  r = []
  i = 0
  while i < len(tags):
    if tags[i] == -1:
      break

    if tags[i] == 0:
      i += 1
    else:
      start, end = i, i
      i += 1
      while i < len(tags):
        if tags[i] == 2:
          end = i
        else:
          break
        i += 1 
      r.append((start, end))
  return r

def evaluate(val_predict, val_targ, tokens, verbose=False):
  val_predict = one_hot_to_labels(val_predict)
  val_target = one_hot_to_labels(val_targ)

  p_names = []
  t_names = []

  num_correct, num_predicted, num_expected = 0, 0, 0
  for i in range(val_predict.shape[0]):
    p_ne = get_named_entities(val_predict[i])
    t_ne = get_named_entities(val_target[i])

    p_names += [tokens[i][ne[0]:ne[1]+1] for ne in p_ne if not tokens[i][ne[1]] is None]
    t_names += [tokens[i][ne[0]:ne[1]+1] for ne in t_ne if not tokens[i][ne[1]] is None]

    num_correct   += len(set(p_ne) & set(t_ne))
    num_predicted += len(p_ne)
    num_expected  += len(t_ne)

  p_names = list(set([' '.join(n) for n in p_names if not n is None]))
  t_names = list(set([' '.join(n) for n in t_names if not n is None]))

  if verbose:
    print('============INCORRECT============')
    for n in [n for n in p_names if not n in t_names]:
      print(n)
    print('============MISSED============')
    for n in [n for n in t_names if not n in p_names]:
      print(n)

  precision = num_correct / float(num_predicted)
  recall = num_correct / float(num_expected)
  correct = num_correct
  incorrect = num_predicted - num_correct
  missed = num_expected - num_correct
  f1 = 2 * precision * recall / (precision + recall)
  return 0, precision, recall, f1, correct, incorrect, missed 
