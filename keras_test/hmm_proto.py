import np
import re
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import sys
import numpy as np

X = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])
Y = np.array([1, 1, 2, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2])

labels = np.array(['O', 'B-NAME', 'I-NAME'])

num_labels = 3
time_steps = 4
vocab_size = 3

num_states = num_labels ** time_steps
transition_mat = np.ones((num_states, num_labels))
emission_mat = np.ones((vocab_size, num_labels))

start = np.zeros((num_states, 1))
start[0,:] = 1
end = np.ones((num_states, 1))

def idx_to_states(idx):
  states = []
  multiplier = num_labels ** (time_steps - 1)
  for i in range(time_steps):
    states.append(int(idx) // int(multiplier))
    idx %= multiplier
    multiplier /= num_labels
  return states 

def states_to_idx(states):
  if len(states) < time_steps:
    raise Exception('Wrong states length.')

  acc = 0
  multiplier = 1
  for s in reversed(states):
    acc += int(multiplier) * int(s)
    multiplier *= num_labels
  return acc

def remove_emails(text):
  """ 
  Removes emails from a string. 
  """
  # The Regex is a simplified version of the one that would account 
  # for all emails, but it is good enough for most cases.
  text = re.sub('\S+@\S+(\.\S+)+', '', text)

  # Sometimes the domain is omitted from the email.
  text = re.sub('\S\S\S+\.\S\S\S+', '', text)
  return text

def remove_urls(text):
  """ 
  Removes urls from a string. 
  """
  regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
  return re.sub(regex, '', text)

def tknize(text):
  text = text.strip().lower()
  text = remove_emails(text)
  text = remove_urls(text)

  special_chars = "ÀÁÂÃÄÅÆÇÈÉÊËÌÍİÎÏÐÑÒÓÔÕÖĞ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿšŽčžŠšČłńężśćŞ"
  chars         = "aaaaaaeceeeeiiiiidnooooogxouuuuypsaaaaaaeceeeeiiiionooooooouuuuypyszczssclnezscs"

  new_text = ""
  for c in text:
    index = special_chars.find(c)
    if index != -1:
      new_text += chars[index]
    else:
      new_text += c

  tkns = re.compile("[^a-zA-Z]+").split(new_text.strip())
  return [t for t in tkns if len(t) > 0]

o, b_name, i_name = {'$UNK': 1}, {'$UNK': 1}, {'$UNK': 1}
def load_gazetteer(name_file, word_file):
  global o, b_name, i_name
  with open(name_file) as f:
    for line in f:
      tkns = tknize(line)
      if len(tkns) == 0:
        continue

      # if not tkns[0] in b_name:
      #   b_name[tkns[0]] = 2 # Laplace smoothing.
      # else:
      #   b_name[tkns[0]] += 1

      for t in tkns: 
        if not t in i_name:
          i_name[t] = 2
        else:
          i_name[t] += 1

  with open(word_file) as f:
    for line in f:
      tkns = line.strip().split(' ')
      for t in tkns: 
        if not t in o:
          o[t] = 2
        else:
          o[t] += 1

  o_count = sum([o[tkn] for tkn in o])
  for tkn in o:
    o[tkn] = float(o[tkn]) / float(o_count)

  b_name_count = sum([b_name[tkn] for tkn in b_name])
  for tkn in b_name:
    b_name[tkn] = float(b_name[tkn]) / float(b_name_count)

  i_name_count = sum([i_name[tkn] for tkn in i_name])
  for tkn in i_name:
    i_name[tkn] = float(i_name[tkn]) / float(i_name_count)

def fit(Y):
  global transition_mat

  states = [0] * time_steps
  for i in range(len(Y)):
    for j in range(50):
      if np.sum(Y[i,j]) == 0:
        break
      y = np.where(Y[i,j] == 1)[0][0]

      idx = states_to_idx(states)
      transition_mat[idx,y] += 1
      states.pop(0) 
      states.append(y) 
  transition_mat /= np.expand_dims(np.sum(transition_mat, axis=1), axis=1)
  transition_mat = np.nan_to_num(transition_mat)

def viterbi(X):
  global o, b_name, i_name
  pointers = np.zeros((len(X), num_states), dtype=int)

  state_probs = start 
  for i in range(len(X)):
    emission = np.ones(num_labels)

    # A name may be composed of more than one token.
    tkns = tknize(X[i])
    if len(tkns) == 0 or bool(re.search(r'[\d\(\)]', X[i])):
      emission[0] = 1
      emission[1] = 0
      emission[2] = 0

    else:
      for key in tkns:
        key = key.strip()
        if key in o:
          emission[0] *= o[key]
        else:
          emission[0] *= o['$UNK']

        if key in i_name:
          emission[1] *= i_name[key]
          emission[2] *= i_name[key]
        else:
          emission[1] *= i_name['$UNK']
          emission[2] *= i_name['$UNK']

    p = state_probs * transition_mat * emission
    state_probs = np.zeros((num_states, 1))
    for s in range(num_states):
      for l in range(num_labels):
        states = idx_to_states(s)
        states.pop(0)
        states.append(l)
        idx = states_to_idx(states)
         
        if p[s,l] > state_probs[idx,:]:
          pointers[i,idx] = s
          state_probs[idx,0] = p[s,l]

  idx = np.argmax(state_probs)
  labels = [] 
  for i in reversed(range(len(X))):
    states = idx_to_states(idx)
    labels.append(states[-1])
    idx = pointers[i,idx]
  return list(reversed(labels))

def predict(X):
  y = np.zeros((len(X), 50, 3))
  for i in range(len(X)):
    tkns = [X[i,j][0] for j in range(50) if not X[i,j] is None]

    # To one hot.
    labels = viterbi(tkns)
    for j, l in enumerate(labels):
      y[i,j,int(l)] = 1
  return y

def flatten(y):
  arr = y[:,:,].argmax(axis=-1).flatten()
  arr[arr > 1] = 1
  return arr

def remove_titles(text):
  text = re.sub('(M\. Sc\.)|(M\.Sc\.)|(MS\. SC\.)|(MS\.SC\.)', '', text)
  text = re.sub('(M\. Ed\.)|(M\.Ed\.)|(M\. ED\.)|(M\.ED\.)', '', text)
  text = re.sub('(sc\. nat\.)|(sc\.nat\.)', '', text)
  text = re.sub('(rer\. nat\.)|(rer\.nat\.)|(rer nat)|(rer\. nat)', '', text)
  text = re.sub('(i\. R\.)', '', text)
  text = re.sub('(PD )|( PD )', '', text)
  text = re.sub('(Sc\. Nat\.)|(Sc\.Nat\.)|(SC\. NAT\.)|(SC\.NAT\.)', '', text)
  text = re.sub('(Sc\. Nat)|(Sc\.Nat)|(SC\. NAT)|(SC\.NAT)', '', text)
  text = re.sub('(MD\.)|(Md\.)|(Md )', '', text)
  text = re.sub('(B\. Sc\.)|(B\.Sc\.)|(BS\. SC\.)|(BS\.SC\.)', '', text)
  text = re.sub('(B\. Sc)|(B\.Sc)|(BS\. SC)|(BS\.SC)', '', text)
  text = re.sub('(Ph\. D\.)|(Ph\.D\.)|(PH\. D\.)|(PH\.D\.)', '', text)
  text = re.sub('(Ph\. D)|(Ph\.D)|(PH\. D)|(PH\.D)', '', text)
  text = re.sub('(Ed\. D\.)|(Ed\.D\.)|(ED\. D\.)|(ED\.D\.)', '', text)
  text = re.sub('(Ed\. D)|(Ed\.D)|(ED\. D)|(ED\.D)', '', text)
  text = re.sub('(M\. S\.)|(M\.S\.)', '', text)
  text = re.sub('(Hon\.)', '', text)
  text = re.sub('(a\.D\.)', '', text)
  text = re.sub('(em\.)', '', text)
  text = re.sub('(apl\.)|(Apl\.)', '', text)
  text = re.sub('(apl\.)|(Apl\.)', '', text)
  text = re.sub('(Prof\.dr\.)', '', text)
  text = re.sub('(Conf\.dr\.)', '', text)
  text = re.sub('(Asist\.dr\.)', '', text)
  text = re.sub('(DR\. )', '', text)
  return text.strip()

from util import Dataset, WordEmbeddings
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

we       = WordEmbeddings('glove-50')
dev      = Dataset(we, 'dataset/dev.txt', max_sentence_len=50, max_token_len=20)
validate = Dataset(we, 'dataset/validate.txt', max_sentence_len=50, max_token_len=20)
# test     = Dataset(we, 'dataset/test.txt', max_sentence_len=50, max_token_len=20)

load_gazetteer('names.txt', 'words.txt')
fit(dev.Y)

val_predict = flatten(predict(dev.T))
val_targ = flatten(dev.Y)

_val_f1 = f1_score(val_targ, val_predict)
_val_recall = recall_score(val_targ, val_predict)
_val_precision = precision_score(val_targ, val_predict)
print(" — f1: %f — precision: %f — recall %f" % (_val_f1, _val_precision, _val_recall))
   
 
tokens = validate.T.flatten()
val_predict = predict(validate.T).argmax(axis=-1).flatten()

names = []
name = []
for i, t in enumerate(tokens):
  if val_predict[i] == 1:
    if len(name) > 0:
      names.append(' '.join(name))
      name = []
    if not t is None:
      name.append(t[0])
  elif val_predict[i] == 2:
    if not t is None:
      name.append(t[0])
  elif len(name) > 0:
    names.append(' '.join(name))
    name = []
if len(name) > 0:
  names.append(' '.join(name))

val_predict = validate.Y.argmax(axis=-1).flatten()
names2 = []
name = []
for i, t in enumerate(tokens):
  if val_predict[i] == 1:
    if len(name) > 0:
      names2.append(' '.join(name))
      name = []
    if not t is None:
      name.append(t[0])
  elif val_predict[i] == 2:
    if not t is None:
      name.append(t[0])
  elif len(name) > 0:
    names2.append(' '.join(name))
    name = []
if len(name) > 0:
  names2.append(' '.join(name))

names = list(set([remove_titles(n) for n in names]))
names2 = list(set(names2))

correct = []
incorrect = []
for n in names:    
  if n in names2:
    correct.append(n)
  else:
    incorrect.append(n)

missed = []
for n in names2:    
  if not n in names:
    missed.append(n)

print('precision:', len(correct) / float(len(names)))
print('recall:',    len(correct) / float(len(names2)))
print('correct:', len(correct))
print('incorrect:', len(incorrect))
print('missed:', len(missed))

for n in missed:
  print(n)
  print(tknize(n))

print('\n\n\n\n')

for n in incorrect:
  print(n)
