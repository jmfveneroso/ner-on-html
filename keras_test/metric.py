import np
np.set_printoptions(threshold=np.nan)

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
      elif len(name) > 0:
        names.append(' '.join(name))
        name = []
    if len(name) > 0:
      names.append(' '.join(name))
  return names

def one_hot_to_labels(Y):
  res = np.zeros((Y.shape[0], Y.shape[1]))
  for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
      res[i,j] = np.argmax(Y[i,j], axis=-1)
  return res

def evaluate(val_predict, val_targ, tokens):
  val_predict = one_hot_to_labels(val_predict)
  val_targ = one_hot_to_labels(val_targ)

  predicted_names = list(set(get_names(val_predict, tokens)))
  target_names = list(set(get_names(val_targ, tokens)))
  
  correct = []
  incorrect = []
  for n in predicted_names:    
    if n in target_names:
      correct.append(n)
    else:
      incorrect.append(n)
      print(n)
  
  print('========================')
  missed = []
  for n in target_names:    
    if not n in predicted_names:
      missed.append(n)
      print(n)
 
  precision = len(correct) / float(len(predicted_names))
  recall = len(correct) / float(len(target_names))
  correct = len(correct)
  incorrect = len(incorrect)
  missed = len(missed)

  return precision, recall, correct, incorrect, missed
