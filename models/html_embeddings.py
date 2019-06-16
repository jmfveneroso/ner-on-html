import tensorflow as tf
from pathlib import Path
from model.char_representations import get_char_embeddings, lstm_char_representations

def get_html_embeddings(html, html_vocab_file):
  with Path(html_vocab_file).open() as f:
    num_html_tags = sum(1 for _ in f) + 1
  
  vocab_html = tf.contrib.lookup.index_table_from_file(
    html_vocab_file, num_oov_buckets=1
  )

  html_tags = tf.slice(html, [0, 0, 0], [-1, -1, 2])
  html_tag_ids = vocab_html.lookup(html_tags)
  return html_tag_ids 

  html_embedding_size = 50
  v = tf.get_variable('html_embeddings', [num_html_tags + 1, html_embedding_size], tf.float32)
  html_embeddings = tf.nn.embedding_lookup(v, html_tag_ids)
  # timesteps = tf.shape(html_tags)[1]
  # html_embeddings = tf.reshape(html_embeddings, [-1, timesteps, html_embedding_size*2])
  html_embeddings = tf.reduce_sum(html_embeddings, axis=-2)
  return html_embeddings

def get_css_embeddings(html, css_chars, css_lengths, char_vocab_file, training=False):
  with Path('data/ner_on_html/vocab.css.txt').open() as f:
    num_css_classes = sum(1 for _ in f) + 1
  
  vocab_css = tf.contrib.lookup.index_table_from_file(
    'data/ner_on_html/vocab.css.txt', num_oov_buckets=1
  )

  css_class = tf.slice(html, [0, 0, 2], [-1, -1, 1])
  css_class_ids = vocab_css.lookup(css_class)
  return css_class_ids

def get_html_representations(html, html_vocab_file, css_chars, css_lengths, char_vocab_file, training=False):
  html_embs = get_html_embeddings(html, html_vocab_file)
  css_embs = get_css_embeddings(
    html, css_chars, css_lengths,
    char_vocab_file, training=training
  )

  html_embs = tf.concat([html_embs, css_embs], axis=-1)
  return html_embs

def get_soft_html_representations(html, html_vocab_file, css_chars, css_lengths, char_vocab_file, training=False):
  with Path(html_vocab_file).open() as f:
    num_html_tags = sum(1 for _ in f) + 1
  
  vocab_html = tf.contrib.lookup.index_table_from_file(
    html_vocab_file, num_oov_buckets=1
  )

  html_tags = tf.slice(html, [0, 0, 0], [-1, -1, 2])
  html_tag_ids = vocab_html.lookup(html_tags)

  html_embedding_size = 25
  v = tf.get_variable('html_embeddings', [num_html_tags + 1, html_embedding_size], tf.float32)
  html_embeddings = tf.nn.embedding_lookup(v, html_tag_ids)
  timesteps = tf.shape(html_tags)[1]
  # html_embeddings = tf.reduce_sum(html_embeddings, axis=-2)
  html_embeddings = tf.reshape(html_embeddings, [-1, timesteps, html_embedding_size*2])

  lstm_size = 25
  char_embedding_size = 50

  char_embeddings = get_char_embeddings(css_chars, char_vocab_file, char_embedding_size, training=training)
  css_embeddings = tf.reduce_mean(char_embeddings, axis=-2)

  return tf.concat([html_embeddings, css_embeddings], axis=-1)
  # return html_embeddings
