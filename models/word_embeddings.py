import tensorflow as tf
import numpy as np
from pathlib import Path

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
  variable = np.vstack([glove, [[0.] * 300]])
  variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
  word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

  return word_embeddings
