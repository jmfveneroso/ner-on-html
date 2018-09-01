import np
import re

def remove_titles(text):
  name = []
  for tkn in text.split(' '):
    titles = ['m\.sc\.','sc\.nat\.','rer\.nat\.','sc\.nat\.','md\.',
      'b\.sc\.', 'bs\.sc\.', 'ph\.d\.', 'ed\.d\.', 'm\.s\.', 'hon\.', 
      'a\.d\.', 'em\.', 'apl\.', 'prof\.', 'prof\.dr\.', 'conf\.dr\.',
      'asist\.dr\.', 'dr\.', 'mr\.', 'mrs\.']

    match = False
    for title in titles:
      if re.match('^' + title + '$', tkn, re.IGNORECASE):
        match = True
        break

    if not match:
      name.append(tkn)
        
  return ' '.join(name)

def get_names(val_predict, sentences):
  names, name = [], []
  for i, tokens in enumerate(sentences):
    for j, t in enumerate(tokens):
      if val_predict[i][j] == 1:
        if len(name) > 0:
          names.append(' '.join(name))
          name = []
        if not t is None:
          name.append(t)
      elif val_predict[i][j] == 2:
        if not t is None:
          name.append(t)
      elif val_predict[i][j] == 3:
        if len(name) > 0 and j+1 < len(tokens):
          if val_predict[i][j+1] == 2:
            name.append(t)
      elif len(name) > 0:
        names.append(' '.join(name))
        name = []

    if len(name) > 0:
      names.append(' '.join(name))
  names = [remove_titles(n) for n in names]
  return [n for n in names if len(n.split(' ')) > 1]

def one_hot_to_labels(Y):
  res = np.zeros((Y.shape[0], Y.shape[1]), dtype=int)
  for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
      if np.sum(Y[i,j]) == 0:
        res[i,j] = -1
      else: 
        res[i,j] = int(np.argmax(Y[i,j], axis=-1))
  return res

def evaluate(val_predict, val_targ, tokens):
  val_predict = one_hot_to_labels(val_predict)
  val_targ = one_hot_to_labels(val_targ)

  confusion_matrix = np.zeros((4,4))

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
      # print(n)
  
  # print('========================')
  missed = []
  for n in target_names:    
    if not n in predicted_names:
      missed.append(n)
      # print(n)

  # print(confusion_matrix)

  accuracy = accuracy / float(total)
  precision = len(correct) / float(len(predicted_names))
  recall = len(correct) / float(len(target_names))
  correct = len(correct)
  incorrect = len(incorrect)
  missed = len(missed)
  f1 = 2 * precision * recall / (precision + recall)

  return accuracy, precision, recall, f1, correct, incorrect, missed 
