import np
from util import load_raw_dataset
from tokenizer import remove_accents, tokenize_text
from metric import evaluate

num_labels   = 3
num_features = 7
time_steps   = 3

num_states = num_labels ** time_steps
transition_mat = np.zeros((num_states, num_labels))
emission_mat = np.zeros((num_features, num_labels))

start = np.zeros((num_states, 1))
start[0,:] = 1
end = np.ones((num_states, 1))

o, i_name, b_name = {'$UNK': 1}, {'$UNK': 1}, {'$UNK': 1}
g_o, g_i_name = {'$UNK': 1}, {'$UNK': 1}

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

def load_gazetteer(name_file, word_file):
  global o, i_name
  with open(name_file) as f:
    count = 0
    for line in f:
      tkns = tokenize_text(line)
      if len(tkns) == 0:
        continue

      for t in tkns:
        t = remove_accents(t)
        if not t in i_name:
          g_i_name[t] = 2
        else:
          g_i_name[t] += 1

  with open(word_file) as f:
    for line in f:
      tkns = line.strip().split(' ')
      for t in tkns: 
        t = remove_accents(t)
        if not t in o:
          g_o[t] = 2
        else:
          g_o[t] += 1

  o_count = sum([o[tkn] for tkn in o])
  for tkn in o:
    g_o[tkn] = float(o[tkn]) / float(o_count)

  i_name_count = sum([i_name[tkn] for tkn in i_name])
  for tkn in i_name:
    g_i_name[tkn] = float(i_name[tkn]) / float(i_name_count)

def fit(X, Y):
  global transition_mat, emission_mat

  label_count = np.ones((num_labels))
  states = [0] * time_steps
  for i in range(len(Y)):
    for j in range(50):
      if np.sum(Y[i,j]) == 0:
        break

      label_count += Y[i,j]
      y = np.where(Y[i,j] == 1)[0][0]

      # f = [int(f) for f in X[i][j][2:9]]
      t = X[i][j][1]
      if y == 0:
        if not t in o:
          o[t] = 2
        else:
          o[t] += 1
      elif y == 1:
        if not t in b_name:
          b_name[t] = 2
        else:
          b_name[t] += 1
      else:
        if not t in i_name:
          i_name[t] = 2
        else:
          i_name[t] += 1

      f = [int(f) for f in X[i][j][2:2+num_features]]
      emission_mat[:,y] += f

      idx = states_to_idx(states)
      transition_mat[idx,y] += 1
      states.pop(0) 
      states.append(y) 

  emission_mat /= label_count                                                                  

  transition_mat /= np.expand_dims(np.sum(transition_mat, axis=1), axis=1)
  transition_mat = np.nan_to_num(transition_mat)
  print(transition_mat)
  print(emission_mat)

  o_count = sum([o[tkn] for tkn in o])
  for tkn in o:
    o[tkn] = float(o[tkn]) / float(o_count)

  b_name_count = sum([b_name[tkn] for tkn in b_name])
  for tkn in b_name:
    b_name[tkn] = float(b_name[tkn]) / float(b_name_count)

  i_name_count = sum([i_name[tkn] for tkn in i_name])
  for tkn in i_name:
    i_name[tkn] = float(i_name[tkn]) / float(i_name_count)

def viterbi(X):
  global o, i_name
  pointers = np.zeros((len(X), num_states), dtype=int)

  state_probs = start 
  for i in range(len(X)):
    emission = np.ones(num_labels)

    x = np.expand_dims(np.array([int(f) for f in X[i][2:2+num_features]]), axis=1)
    emission = x * emission_mat # Positives.
    emission += -(x-1) * (1 - emission_mat) # Negatives.
    emission = np.prod(emission, axis=0)
    key = X[i][1]
    if key in o:
      emission[0] *= o[key]
    else:
      emission[0] *= o['$UNK']

    if key in b_name:
      emission[1] *= b_name[key]
    else:
      emission[1] *= b_name['$UNK']

    if key in i_name:
      emission[2] *= i_name[key]
    else:
      emission[2] *= i_name['$UNK']

    if key in g_o:
      emission[0] *= g_o[key]
    else:
      emission[0] *= g_o['$UNK']

    # emission[0] *= g_i_name['$UNK']
    # emission[0] *= g_i_name['$UNK']
    if key in g_i_name:
      emission[1] *= g_i_name[key]
      emission[2] *= g_i_name[key]
    else:
      emission[1] *= g_i_name['$UNK']
      emission[2] *= g_i_name['$UNK']

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
    labels = viterbi(X[i])

    # To one hot.
    for j, l in enumerate(labels):
      y[i,j,int(l)] = 1
  return y

X2, Y2, T2 = load_raw_dataset('dataset/dev.txt')
# X, Y, T    = load_raw_dataset('dataset/001.txt')
# X, Y, T    = load_raw_dataset('dataset/test.txt')
X, Y, T    = load_raw_dataset('dataset/validate.txt')

load_gazetteer('names.txt', 'words.txt')
fit(X2, Y2)

p, r, c, i, m = evaluate(predict(X), Y, T)

print('Precision:', p)
print('Recall:', r)
print('Correct:', c)
print('Incorrect:', i)
print('Missed:', m)
