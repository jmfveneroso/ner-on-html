import re
import functools
import json
import math
import random
import tensorflow as tf
from pathlib import Path

def get_sentences(f): 
  filename = f.parts[-1] 
  with f.open('r', encoding="utf-8") as f:
    sentences = f.read().strip().split('\n\n')
    sentences = [[t.split() for t in s.split('\n')] for s in sentences if len(s) > 0] 
    return sentences

def split_array(arr, separator_fn):
  arrays = [[]]
  for i, el in enumerate(arr):
    if separator_fn(el):
      arrays.append([])
    else:
      arrays[-1].append(el)
  return [a for a in arrays if len(a) > 0]

def join_arrays(arr, separator):
  res = []
  for i, el in enumerate(arr):
    res += el + [separator]
  return res

def pad_array(arr, padding, max_len):
  return arr + [padding] * (max_len - len(arr))

def prepare_dataset(sentences, mode='sentences', label_col=3, feature_cols=[], training=False):
  sentences = [
    [
      [ t[0], t[label_col], [ t[f] for f in feature_cols if len(t) > f ] ]
      for t in s
    ]
    for s in sentences
  ] 

  if mode == 'sentences':
    sentences = [s for s in sentences if s[0][0] != '-DOCSTART-']
    return sentences

  else:
    separator_fn = lambda el : el[0][0] == '-DOCSTART-'
    documents = split_array(sentences, separator_fn)

    # Shuffle sentences inside document.
    if training:
      for i, d in enumerate(documents):
        random.shuffle(documents[i])
 
    eos = ['EOS', 'O', ['0'] * len(feature_cols)]
    documents = [join_arrays(d, eos) for d in documents]

    if mode == 'documents':
      return documents

    elif mode == 'batch':
      n = 10
      sentences = []
      for d in documents:
        cur_sentence = []
        eos_count = 0 
        for t in d:
          if t[0] == 'EOS':
            eos_count += 1

          if eos_count > n:
            eos_count = 0
            sentences.append(cur_sentence)
            cur_sentence = []
          else:
            cur_sentence.append(t)

        if len(cur_sentence) > 0:
          sentences.append(cur_sentence)
      return sentences

    else: 
      raise Exception('Invalid mode.') 

class DL:
  __instance = None
  def __new__(cls, *args, **kwargs):
    if not DL.__instance:
      DL.__instance = DL.__DataLoader(*args, **kwargs)
    return DL.__instance 

  def __getattr__(self, name):
    return getattr(self.instance, name)

  class __DataLoader:
    def __init__(self):
      self.params = {
        'label_col': 3,
        'batch_size': 1,
        'buffer': 15000,
        'datadir': 'data/conll2003',
        'fulldoc': True,
        'splitsentence': False,
      }

    def __str__(self):
      return repr(self)

    def save_params(self):
      with Path(self.params['datadir'], 'params.json').open('w') as f:
        json.dump(self.params, f, indent=2, sort_keys=True)

    def parse_sentence(self, sentence):
      # Encode in Bytes for Tensorflow.
      words = [s[0] for s in sentence]
      tags = [s[1].encode() for s in sentence]
 
      # Chars.
      chars = [[c.encode() for c in w] for w in words]
      lengths = [len(c) for c in chars]
      chars = [pad_array(c, b'<pad>', max(lengths)) for c in chars]
      
      # HTML features.
      html_features = [[f.encode() for f in s[2]] for s in sentence]
      html_features = [pad_array(f, b'<pad>', 3) for f in html_features]

      # CSS Chars.
      css_chars = [[c.encode() for c in f[2].decode()] for f in html_features]
      css_lengths = [len(c) for c in css_chars]
      css_chars = [pad_array(c, b'<pad>', max(css_lengths)) for c in css_chars]
      
      words = [s[0].encode() for s in sentence]    
      return ((words, len(words)), (chars, lengths), html_features, (css_chars, css_lengths)), tags
      
    def generator_fn(self, filename, training=False):
      sentences = get_sentences(Path(self.params['datadir'], filename))

      mode = 'sentences'
      label_col = 3
      feature_cols = []
      if self.params['datadir'] == 'data/ner_on_html':
        label_col = 1
        feature_cols = [13, 14, 15]
        for i, s in enumerate(sentences):
          for j, t in enumerate(s):
            if t[0] != '-DOCSTART-':
              sentences[i][j] = t[:13] + t[13].split('.') + t[14:]

      if self.params['fulldoc']:
        mode = 'documents'
      if self.params['splitsentence']:
        mode = 'batch'
      sentences = prepare_dataset(sentences, mode=mode, label_col=label_col, feature_cols=feature_cols, training=training)

      for s in sentences:
        yield self.parse_sentence(s)
          
    def input_fn(self, filename, training=False):
      shapes = (
       (([None], ()),           # (words, nwords)
       ([None, None], [None]),  # (chars, nchars)  
       [None, None],            # html_features
       ([None, None], [None])), # (css_chars, css_lengths)  
       [None]                   # tags
      )
    
      types = (
        ((tf.string, tf.int32),
        (tf.string, tf.int32),  
        tf.string,
        (tf.string, tf.int32)),  
        tf.string
      )
    
      defaults = (
        (('<pad>', 0),
        ('<pad>', 0), 
        '<pad>',
        ('<pad>', 0)), 
        'O'
      )
    
      dataset = tf.data.Dataset.from_generator(
        functools.partial(self.generator_fn, filename, training=training),
        output_types=types, output_shapes=shapes
      )
    
      if training:
        dataset = dataset.shuffle(self.params['buffer'])
   
      return dataset.padded_batch(self.params['batch_size'], shapes, defaults)

    def get_doc(self, filename, doc):
      p1, p2 = self.params['fulldoc'], self.params['splitsentence']
      self.params['fulldoc'], self.params['splitsentence'] = True, False
      features, labels = [(f, l) for (f, l) in DL().generator_fn(filename)][doc]
      self.params['fulldoc'], self.params['splitsentence'] = p1, p2
      (words, nwords), (chars, nchars), html, (css_chars, css_lengths) = features
      features = ([words], [nwords]), ([chars], [nchars])
      return features, [labels]

    def set_params(self, params=None):
      params = params if params is not None else {}
      self.params.update(params)

      if self.params['datadir'] == 'data/conll2003':
        self.params['label_col'] = 3
      elif self.params['datadir'] == 'data/ner_on_html':
        self.params['label_col'] = 1

      if self.params['fulldoc']:
        self.params['batch_size'] = 1

class Conll2003:
  def __init__(self, params=None):
    self.params = {
      'label_col': 3,
      'batch_size': 1,
      'buffer': 15000,
      'datadir': 'data/conll2003',
      'fulldoc': True,
      'splitsentence': False,
    }
    self.set_params(params)

  def set_params(self, params=None):
    params = params if params is not None else {}
    self.params.update(params)

  def parse_sentence(self, sentence):
    # Encode in Bytes for Tensorflow.
    words = [s[0] for s in sentence]
    tags = [s[1].encode() for s in sentence]

    # Chars.
    chars = [[c.encode() for c in w] for w in words]
    lengths = [len(c) for c in chars]
    chars = [pad_array(c, b'<pad>', max(lengths)) for c in chars]
    
    words = [s[0].encode() for s in sentence]    
    return ((words, len(words)), (chars, lengths)), (), tags
    
  def generator_fn(self, filename, training=False):
    sentences = get_sentences(Path(self.params['datadir'], filename))

    mode = 'sentences'
    label_col = 3
    feature_cols = []
    if self.params['datadir'] == 'data/ner_on_html':
      label_col = 1
      feature_cols = [13, 14, 15]
      for i, s in enumerate(sentences):
        for j, t in enumerate(s):
          if t[0] != '-DOCSTART-':
            sentences[i][j] = t[:13] + t[13].split('.') + t[14:]

    if self.params['fulldoc']:
      mode = 'documents'
    if self.params['splitsentence']:
      mode = 'batch'
    sentences = prepare_dataset(sentences, mode=mode, label_col=label_col, feature_cols=feature_cols, training=training)

    for s in sentences:
      yield self.parse_sentence(s)
        
  def input_fn(self, filename, training=False):
    shapes = (
     (([None], ()),           # (words, nwords)
     ([None, None], [None]),  # (chars, nchars)  
     [None, None],            # html_features
     ([None, None], [None])), # (css_chars, css_lengths)  
     [None]                   # tags
    )
  
    types = (
      ((tf.string, tf.int32),
      (tf.string, tf.int32),  
      tf.string,
      (tf.string, tf.int32)),  
      tf.string
    )
  
    defaults = (
      (('<pad>', 0),
      ('<pad>', 0), 
      '<pad>',
      ('<pad>', 0)), 
      'O'
    )
  
    dataset = tf.data.Dataset.from_generator(
      functools.partial(self.generator_fn, filename, training=training),
      output_types=types, output_shapes=shapes
    )
  
    if training:
      dataset = dataset.shuffle(self.params['buffer'])
 
    return dataset.padded_batch(self.params['batch_size'], shapes, defaults)

  def get_doc(self, filename, doc):
    p1, p2 = self.params['fulldoc'], self.params['splitsentence']
    self.params['fulldoc'], self.params['splitsentence'] = True, False
    features, labels = [(f, l) for (f, l) in DL().generator_fn(filename)][doc]
    self.params['fulldoc'], self.params['splitsentence'] = p1, p2
    (words, nwords), (chars, nchars), html, (css_chars, css_lengths) = features
    features = ([words], [nwords]), ([chars], [nchars])
    return features, [labels]

class NerOnHtml:
  def __init__(self, params=None):
    self.params = {
      'batch_size': 1,
      'datadir': 'data/ner_on_html',
      'dataset_mode': ('sentences', 'document', 'batch')[0]
    }
    self.set_params(params)

  def set_params(self, params=None):
    params = params if params is not None else {}
    self.params.update(params)

  def parse_sentence(self, sentence):
    # Encode in Bytes for Tensorflow.
    words = [s[0] for s in sentence]
    tags = [s[1].encode() for s in sentence]

    # Chars.
    chars = [[c.encode() for c in w] for w in words]
    lengths = [len(c) for c in chars]
    chars = [pad_array(c, b'<pad>', max(lengths)) for c in chars]
    
    # Feature vector. 
    # features = [[float(f) for f in s[2][:2] + s[2][4:10]] for s in sentence]
    features = [[float(f) for f in s[2][:2] + s[2][4:9]] for s in sentence]

    # HTML features.
    html_features = [[f.encode() for f in s[2][10:]] for s in sentence]
    html_features = [pad_array(f, b'<pad>', 3) for f in html_features]

    # CSS Chars.
    css_chars = [[c.encode() for c in f[2].decode()] for f in html_features]
    css_lengths = [len(c) for c in css_chars]
    css_chars = [pad_array(c, b'<pad>', max(css_lengths)) for c in css_chars]
    
    words = [s[0].encode() for s in sentence]    
    return (
      (
        ( (words, len(words)), (chars, lengths) ),
        ( features, html_features, (css_chars, css_lengths) ), 
      ),
      tags
    )
    
  def generator_fn(self, filename, training=False):
    sentences = get_sentences(Path(self.params['datadir'], filename))

    # Split HTML tag feature.
    for i, s in enumerate(sentences):
      for j, t in enumerate(s):
        if t[0] != '-DOCSTART-':
          sentences[i][j] = t[:13] + t[13].split('.') + t[14:]

    sentences = prepare_dataset(
      sentences, mode=self.params['dataset_mode'], 
      label_col=1, feature_cols=range(3, 15), 
      training=training
    )

    for s in sentences:
      yield self.parse_sentence(s)
        
  def input_fn(self, filename, training=False):
    shapes = (
      (
        (
          ([None], ()), # (words, nwords)
          ([None, None], [None]), # (chars, nchars)  
        ),
        (
          [None, None], # features
          [None, None], # html_features
          ([None, None], [None]), # (css_chars, css_lengths)  
        )
      ),
      [None] # tags
    )
  
    types = (
      (
        (
          (tf.string, tf.int32),
          (tf.string, tf.int32),  
        ),
        (
          tf.float32,
          tf.string,
          (tf.string, tf.int32)
        )
      ),  
      tf.string
    )
  
    defaults = (
      (
        (
          ('<pad>', 0),
          ('<pad>', 0), 
        ),
        (
          0.0,
          '<pad>',
          ('<pad>', 0)
        )
      ), 
      'O'
    )
  
    dataset = tf.data.Dataset.from_generator(
      functools.partial(self.generator_fn, filename, training=training),
      output_types=types, output_shapes=shapes
    )
  
    if training:
      dataset = dataset.shuffle(15000)
 
    return dataset.padded_batch(self.params['batch_size'], shapes, defaults)

class Conll2003Person:
  def __init__(self, params=None):
    self.params = {
      'batch_size': 1,
      'datadir': 'data/conll2003_person',
      'dataset_mode': ('sentences', 'document', 'batch')[0]
    }
    self.set_params(params)

  def set_params(self, params=None):
    params = params if params is not None else {}
    self.params.update(params)

  def parse_sentence(self, sentence):
    # Encode in Bytes for Tensorflow.
    words = [s[0] for s in sentence]
    tags = [s[1].encode() for s in sentence]

    # Chars.
    chars = [[c.encode() for c in w] for w in words]
    lengths = [len(c) for c in chars]
    chars = [pad_array(c, b'<pad>', max(lengths)) for c in chars]
    
    words = [s[0].encode() for s in sentence]    
    return (
      (
        (words, len(words)), (chars, lengths)
      ),
      tags
    )
    
  def generator_fn(self, filename, training=False):
    sentences = get_sentences(Path(self.params['datadir'], filename))

    sentences = prepare_dataset(
      sentences, mode=self.params['dataset_mode'], 
      label_col=3, training=training
    )

    for s in sentences:
      yield self.parse_sentence(s)
        
  def input_fn(self, filename, training=False):
    shapes = (
      (
        ([None], ()), # (words, nwords)
        ([None, None], [None]), # (chars, nchars)  
      ),
      [None] # tags
    )
  
    types = (
      (
        (tf.string, tf.int32),
        (tf.string, tf.int32),  
      ),  
      tf.string
    )
  
    defaults = (
      (
        ('<pad>', 0),
        ('<pad>', 0), 
      ), 
      'O'
    )
  
    dataset = tf.data.Dataset.from_generator(
      functools.partial(self.generator_fn, filename, training=training),
      output_types=types, output_shapes=shapes
    )
  
    if training:
      dataset = dataset.shuffle(15000)
 
    return dataset.padded_batch(self.params['batch_size'], shapes, defaults)
