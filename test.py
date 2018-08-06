# !/usr/bin/python

import sys
import os
import collections
import string
from chunker import NamedEntityChunker
from nltk import pos_tag, word_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
import nltk

directory = 'gmb-2.2.0/'

def to_conll_iob(annotated_sentence):
  """
  `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
  Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
  to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
  """
  proper_iob_tokens = []
  for idx, annotated_token in enumerate(annotated_sentence):
    tag, word, ner = annotated_token
 
    if ner != 'O':
      if idx == 0:
        ner = "B-" + ner
      elif annotated_sentence[idx - 1][2] == ner:
        ner = "I-" + ner
      else:
        ner = "B-" + ner
    proper_iob_tokens.append((tag, word, ner))
  return proper_iob_tokens

def reader():
  with open(directory + 'ner_dataset.csv', 'r') as f:
    f.readline() # Ignore first line.
    
    tokens = []
    for line in f:
      data = line.strip().split(',')
      if data[0].startswith('Sentence: ') and len(tokens) > 0:
        conll_tokens = to_conll_iob(tokens)
        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
        # Because the classfier expects a tuple as input, first item input, second the class
        yield [((w, t), iob) for w, t, iob in conll_tokens]

        tokens = []
      tokens.append((data[1], data[2], data[3]))

data = list(reader())
print len(data)

training_samples = data[:int(len(data) * 0.9)]
test_samples = data[int(len(data) * 0.9):]
 
print "#training samples = %s" % len(training_samples)    # training samples = 55809
print "#test samples = %s" % len(test_samples)

chunker = NamedEntityChunker(training_samples[:2000])
print chunker.parse(pos_tag(word_tokenize("I'm going to Germany this Monday.")))

test = [conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]]
score = chunker.evaluate(test)
print score.accuracy() 

# quit()
# print '-----------------'
# print len(score.correct())
# for i in score.correct():
#   print i
# print '-----------------'
# print len(score.incorrect())
# for i in score.incorrect():
#   print i
# print '-----------------'
# print len(score.missed())
# for i in score.missed():
#   print i
# print '-----------------'


