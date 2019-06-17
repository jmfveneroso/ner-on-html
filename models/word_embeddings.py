import tensorflow as tf
import numpy as np
from pathlib import Path

# def trainable_embs(words, words_vocab_file):
#   vocab_size = 0
#   with Path(words_vocab_file).open() as f:
#     indices = [idx for idx, tag in enumerate(f)]
#     vocab_size = len(indices) + 1
# 
#   vocab_words = tf.contrib.lookup.index_table_from_file(
#     words_vocab_file, num_oov_buckets=1
#   )
# 
#   word_ids = vocab_words.lookup(words)
#   variable = np.vstack([glove, [[0.] * 300]])
#   variable = tf.Variable(variable, dtype=tf.float32, trainable=True)
#   word_embeddings = tf.nn.embedding_lookup(variable, word_ids)
# 
#   return word_embeddings
#   
# 
#   return tf.one_hot(word_ids, vocab_size)

def one_hot_embs(words, words_vocab_file):
  vocab_size = 0
  with Path(words_vocab_file).open() as f:
    indices = [idx for idx, tag in enumerate(f)]
    vocab_size = len(indices) + 1

  vocab_words = tf.contrib.lookup.index_table_from_file(
    words_vocab_file, num_oov_buckets=1
  )

  word_ids = vocab_words.lookup(words)
  return tf.one_hot(word_ids, vocab_size)

def glove(words, words_vocab_file, glove_file):
  vocab_words = tf.contrib.lookup.index_table_from_file(
    words_vocab_file, num_oov_buckets=1
  )

  word_ids = vocab_words.lookup(words)
  glove = np.load(glove_file)['embeddings']
  print(glove.shape)
  variable = np.vstack([glove, [[0.] * 300]])
  variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
  word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

  return word_embeddings

def word2vec(words, words_vocab_file, w2v_file):
  vocab_words = tf.contrib.lookup.index_table_from_file(
    words_vocab_file, num_oov_buckets=1
  )

  word_ids = vocab_words.lookup(words)
  w2v = np.load(w2v_file)['embeddings']
  variable = np.vstack([w2v, [[0.] * 300]])
  variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
  word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

  return word_embeddings

def elmo(words, nwords):
  elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
  
  word_embeddings = elmo(
    inputs={
      "tokens": words,
      "sequence_len": nwords
    },
    signature="tokens",
    as_dict=True
  )["elmo"]

  return word_embeddings
