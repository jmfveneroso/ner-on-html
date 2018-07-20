import os
import collections

import string
from chunker import NamedEntityChunker

from nltk import pos_tag, word_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
 
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

def read_gmb(corpus_root):
  for root, dirs, files in os.walk(corpus_root):
    for filename in files:
      if filename.endswith(".tags"):
        with open(os.path.join(root, filename), 'rb') as file_handle:
          file_content = file_handle.read().decode('utf-8').strip()
          annotated_sentences = file_content.split('\n\n')
          for annotated_sentence in annotated_sentences:
            annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
 
            standard_form_tokens = []
 
            for idx, annotated_token in enumerate(annotated_tokens):
              annotations = annotated_token.split('\t')
              word, tag, ner = annotations[0], annotations[1], annotations[3]
 
              if ner != 'O':
                ner = ner.split('-')[0]
 
              if tag in ('LQU', 'RQU'):   # Make it NLTK compatible
                tag = "``"
 
              standard_form_tokens.append((word, tag, ner))
 
            conll_tokens = to_conll_iob(standard_form_tokens)
 
            # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
            # Because the classfier expects a tuple as input, first item input, second the class
            yield [((w, t), iob) for w, t, iob in conll_tokens]

reader = read_gmb('gmb-2.2.0')

# for x in reader:
#   print x
# quit()

data = list(reader)
training_samples = data[:int(len(data) * 0.9)]
test_samples = data[int(len(data) * 0.9):]
 
# print "#training samples = %s" % len(training_samples)    # training samples = 55809
# print "#test samples = %s" % len(test_samples)

chunker = NamedEntityChunker(training_samples[:2000])

print chunker.parse(pos_tag(word_tokenize("I'm going to Germany this Monday.")))

test = [conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]]
score = chunker.evaluate(test)
print score.accuracy() 

quit()
print '-----------------'
print len(score.correct())
for i in score.correct():
  print i
print '-----------------'
print len(score.incorrect())
for i in score.incorrect():
  print i
print '-----------------'
print len(score.missed())
for i in score.missed():
  print i
print '-----------------'
