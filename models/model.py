import math
import numpy as np 
import tensorflow as tf
from pathlib import Path
from models.char_representations import get_char_representations, get_char_embeddings
from models.attention import attention, exact_attention, pos_embeddings, normalize, multihead_attention
from models.word_embeddings import glove, elmo, word2vec, one_hot_embs
from models.html_embeddings import get_html_representations, get_soft_html_representations

class SequenceModel:
  def __init__(self, params=None):
    self.params = {
      # Static configurations.
      'datadir': 'data',
      # General configurations.
      'lstm_size': 200,
      'decoder': 'crf', # crf, logits.
      'char_representation': 'cnn',
      'word_embeddings': 'glove', # glove, elmo. TODO: bert.
      'model': 'bi_lstm_crf', # bi_lstm_crf, html_attention, self_attention, transformer, crf
      'use_features': False, 
      'f_score_alpha': 0.5,
    }
    params = params if params is not None else {}
    self.params.update(params)

    self.params['words']     = str(Path(self.params['datadir'], 'vocab.words.txt'))
    self.params['chars']     = str(Path(self.params['datadir'], 'vocab.chars.txt'))
    self.params['tags' ]     = str(Path(self.params['datadir'], 'vocab.tags.txt'))
    self.params['html_tags'] = str(Path(self.params['datadir'], 'vocab.htmls.txt'))
    self.params['glove']     = str(Path(self.params['datadir'], 'glove.npz'))
    self.params['word2vec']  = str(Path(self.params['datadir'], 'word2vec.npz'))

    with Path(self.params['tags']).open() as f:
      indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
      self.num_tags = len(indices) + 1
  
  def dropout(self, x):
    return tf.layers.dropout(x, rate=0.5, training=self.training)

  def create_placeholders(self):
    with tf.name_scope('inputs'):
      self.words         = tf.placeholder(tf.string,  shape=(None, None),       name='words'       )
      self.nwords        = tf.placeholder(tf.int32,   shape=(None,),            name='nwords'      )
      self.chars         = tf.placeholder(tf.string,  shape=(None, None, None), name='chars'       )
      self.nchars        = tf.placeholder(tf.int32,   shape=(None, None),       name='nchars'      )
      self.features      = tf.placeholder(tf.float32, shape=(None, None, 7),    name='features'    )
      self.html          = tf.placeholder(tf.string,  shape=(None, None, None), name='html'        )
      self.css_chars     = tf.placeholder(tf.string,  shape=(None, None, None), name='css_chars'   )
      self.css_lengths   = tf.placeholder(tf.int32,   shape=(None, None),       name='css_lengths' )
      self.labels        = tf.placeholder(tf.string,  shape=(None, None),       name='labels'      )
      self.training      = tf.placeholder(tf.bool,    shape=(),                 name='training'    )
      self.learning_rate = tf.placeholder_with_default(
        0.00001, shape=(), name='learning_rate'      
      )
 
  def lstm(self, x, lstm_size, var_scope='lstm'):
    with tf.variable_scope(var_scope):
      lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_size)
      lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_size)
      
      (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(
        lstm_cell_fw, lstm_cell_bw, x,
        dtype=tf.float32,
        sequence_length=self.nwords
      )
  
      output = tf.concat([output_fw, output_bw], axis=-1)
      output = self.dropout(output)
      return output

  def output_layer(self, x):
    with tf.name_scope('output'):
      vocab_tags = tf.contrib.lookup.index_table_from_file(self.params['tags'])
      tags = vocab_tags.lookup(self.labels)
      if self.params['decoder'] == 'crf':
        logits = tf.layers.dense(x, self.num_tags)

        transition_matrix = tf.get_variable("crf", [self.num_tags, self.num_tags], dtype=tf.float32)
        pred_ids, _ = tf.contrib.crf.crf_decode(logits, transition_matrix, self.nwords)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, tags, self.nwords, transition_matrix)
        loss = tf.reduce_mean(-log_likelihood, name='loss')
      else:
        if self.params['loss'] == 'f1':
          logits = tf.layers.dense(x, 2)
          pred_ids = tf.argmax(logits, axis=-1)*2

          probs = tf.nn.softmax(logits)
          
          a_tilde = tf.squeeze(tf.slice(probs, [0, 0, 1], [-1, -1, 1]), axis=-1, name='a_tilde_prev')
          mask = tf.cast(tf.equal(tags, 2), dtype=tf.float32, name='mask')
          masked_a = tf.multiply(a_tilde, mask, name='masked_a')
          a_tilde = tf.reduce_sum(masked_a, axis=-1, name='a_tilde')

          n_pos = tf.reduce_sum(tf.cast(tags, dtype=tf.float32), axis=-1, name='n_pos')
          m_pos_tilde = tf.squeeze(tf.slice(probs, [0, 0, 1], [-1, -1, 1]), axis=-1)
          m_pos_tilde = tf.reduce_sum(m_pos_tilde, axis=-1, name='m_pos_tilde') 

          alpha = float(self.params['f_score_alpha'])
          divisor = alpha * n_pos + (1 - alpha) * m_pos_tilde
          loss = tf.divide(a_tilde, divisor)
          loss = tf.reduce_mean(-loss, name='loss')

        else:
          logits = tf.layers.dense(x, self.num_tags)
          pred_ids = tf.argmax(logits, axis=-1)

          labels = tf.one_hot(tags, self.num_tags)
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits
          ), name='loss')

      correct = tf.equal(tf.to_int64(pred_ids), tags)
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
      train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

      reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
        self.params['tags']
      )
      pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))  

  def crf(self, word_embs='glove', char_embs='cnn', use_features=False):
    if word_embs == 'elmo':
      word_embs = elmo(self.words, self.nwords)
    elif word_embs == 'glove':
      word_embs = glove(self.words, self.params['words'], self.params['glove'])
    elif word_embs == 'word2vec':
      word_embs = word2vec(self.words, self.params['words'], self.params['word2vec'])
    elif word_embs == 'one_hot':
      word_embs = one_hot_embs(self.words, self.params['words'])
    else:
      raise Exception('No word embeddings were selected.')

    embs = [word_embs]

    if use_features: 
      embs.append(self.features)

    embs = tf.concat(embs, axis=-1)
    # embs = self.dropout(embs)
    return embs

  def lstm_crf(self, word_embs='glove', char_embs='cnn', use_features=False):
    if word_embs == 'elmo':
      word_embs = elmo(self.words, self.nwords)
    elif word_embs == 'glove':
      word_embs = glove(self.words, self.params['words'], self.params['glove'])
    elif word_embs == 'word2vec':
      word_embs = word2vec(self.words, self.params['words'], self.params['word2vec'])
    elif word_embs == 'one_hot':
      word_embs = one_hot_embs(self.words, self.params['words'])
    else:
      raise Exception('No word embeddings were selected.')

    word_embs = self.dropout(word_embs)
    embs = [word_embs]
    if char_embs in ['cnn', 'lstm']:
      char_embs = get_char_representations(
        self.chars, self.nchars, 
        self.params['chars'], mode=char_embs,
        training=self.training
      )
      char_embs = self.dropout(char_embs)
      embs.append(char_embs)

    if use_features: 
      embs.append(self.features)

    embs = tf.concat(embs, axis=-1)
    return self.lstm(embs, self.params['lstm_size'])

  def self_attention(self, num_heads=1, residual='concat', queries_eq_keys=False):
    word_embs = glove(self.words, self.params['words'], self.params['glove'])
    char_embs = get_char_representations(
      self.chars, self.nchars, 
      self.params['chars'], mode='lstm',
      training=self.training
    )
    html_embs = get_soft_html_representations(
      self.html, self.params['html_tags'],
      self.css_chars, self.css_lengths,
      self.params['chars'], training=self.training
    )

    embs = tf.concat([word_embs, char_embs, html_embs], axis=-1)
    embs = self.dropout(embs)
    output = self.lstm(embs, self.params['lstm_size'])
    output = self.dropout(output)

    return attention(
      output, output, num_heads,
      residual=residual, queries_eq_keys=queries_eq_keys,
      training=self.training
    )

  def html_attention(self, word_embs='glove', char_embs='cnn', num_heads=2, residual='add'):
    if word_embs == 'elmo':
      word_embs = elmo(self.words, self.nwords)
    elif word_embs == 'glove':
      word_embs = glove(self.words, self.params['words'], self.params['glove'])
    elif word_embs == 'word2vec':
      word_embs = word2vec(self.words, self.params['words'], self.params['word2vec'])
    else:
      raise Exception('No word embeddings were selected.')

    char_embs = get_char_representations(
      self.chars, self.nchars, 
      self.params['chars'], mode=char_embs,
      training=self.training
    )

    embs = tf.concat([word_embs, char_embs], axis=-1)
    embs = self.dropout(embs)
    output = self.lstm(embs, self.params['lstm_size'])
    output = self.dropout(output)

    html_embs = get_html_representations(
      self.html, self.params['html_tags'],
      self.css_chars, self.css_lengths,
      self.params['chars'], training=self.training
    )

    return exact_attention(html_embs, html_embs, output, residual=residual, training=self.training)

  def create(self):
    self.create_placeholders()

    model = self.params['model']
    if model == 'bi_lstm_crf':
      output = self.lstm_crf(word_embs=self.params['word_embeddings'], char_embs=self.params['char_representation'], use_features=self.params['use_features'])
    elif model == 'html_attention':
      output = self.html_attention(word_embs=self.params['word_embeddings'], char_embs=self.params['char_representation'])
    elif model == 'self_attention':
      output = self.self_attention()
    elif model == 'transformer':
      output = self.transformer()
    elif model == 'crf':
      output = self.crf(word_embs=self.params['word_embeddings'], char_embs=self.params['char_representation'], use_features=self.params['use_features'])
    else:
      raise Exception('Model does not exist.')

    self.output_layer(output)
