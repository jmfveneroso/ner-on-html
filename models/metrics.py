from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
import tensorflow as tf
import numpy as np 

def get_named_entities(tags):
  try:
    tags = [t.decode("utf-8") for t in tags]
  except AttributeError:
    pass

  r = []
  i = 0
  while i < len(tags):
    if tags[i] == 'O':
      i += 1
    else:
      tag_type = tags[i][2:]
      expected_tag = 'I-' + tag_type
      start, end = i, i
      i += 1
      while i < len(tags):
        if tags[i] == expected_tag:
          end = i
        else:
          break
        i += 1
      r.append((start, end, tag_type))
  return r

def evaluate(val_predict, val_target, tokens, verbose=False):
  correct_count, total = 0, 0
  num_correct, num_predicted, num_expected = 0, 0, 0
  for i in range(len(val_predict)):
    correct_count += sum(p == t for p, t in zip(val_predict[i], val_target[i]))
    total += len(val_target[i])
    p_ne = get_named_entities(val_predict[i])
    t_ne = get_named_entities(val_target[i])

    num_correct   += len(set(p_ne) & set(t_ne))
    num_predicted += len(p_ne)
    num_expected  += len(t_ne)

  precision, recall, f1, accuracy = 0, 0, 0, 0

  if num_predicted > 0:
    precision = num_correct / float(num_predicted)

  if num_expected > 0:
    recall = num_correct / float(num_expected)

  if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)

  if total > 0:
    accuracy = correct_count / float(total)

  correct = num_correct
  incorrect = num_predicted - num_correct
  missed = num_expected - num_correct

  return {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'correct': correct,
    'incorrect': incorrect,
    'missed': missed
  }
