import tensorflow as tf
from pathlib import Path
from six.moves import reduce

def cnn_char_representations(t, weights, filters, kernel_size):
  shape = tf.shape(t)
  ndims = t.shape.ndims
  dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
  dim2 = shape[-2]
  dim3 = t.shape[-1]

  # Reshape weights
  weights = tf.reshape(weights, shape=[dim1, dim2, 1])
  weights = tf.to_float(weights)

  # Reshape input and apply weights
  flat_shape = [dim1, dim2, dim3]
  t = tf.reshape(t, shape=flat_shape)
  t *= weights

  # Apply convolution
  t_conv = tf.layers.conv1d(t, filters, kernel_size, padding='same')
  t_conv *= weights

  # Reduce max -- set to zero if all padded
  t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
  t_max = tf.reduce_max(t_conv, axis=-2)

  # Reshape the output
  final_shape = [shape[i] for i in range(ndims-2)] + [filters]
  t_max = tf.reshape(t_max, shape=final_shape)
  return t_max

def lstm_char_representations(char_embeddings, nchars, lstm_size, char_embedding_size, scope='lstm_chars'):
  with tf.variable_scope(scope):
    dim_words = tf.shape(char_embeddings)[1]
    dim_chars = tf.shape(char_embeddings)[2]

    t = tf.reshape(char_embeddings, [-1, dim_chars, char_embedding_size])

    lstm_cell_fw_c = tf.nn.rnn_cell.LSTMCell(lstm_size)
    lstm_cell_bw_c = tf.nn.rnn_cell.LSTMCell(lstm_size)
    
    (_, _), (output_fw, output_bw) = tf.nn.bidirectional_dynamic_rnn(
      lstm_cell_fw_c, lstm_cell_bw_c, t,
      dtype=tf.float32,
      sequence_length=tf.reshape(nchars, [-1]),
    )

    # output_fw[0] is the cell state and output_fw[1] is the hidden state.
    output = tf.concat([output_fw[1], output_bw[1]], axis=-1)
    return tf.reshape(output, [-1, dim_words, 2 * lstm_size])

def get_char_embeddings(chars, char_vocab_file, char_embedding_size, training=False):
  with tf.variable_scope("char_embeddings", reuse=tf.AUTO_REUSE):
    with Path(char_vocab_file).open() as f:
      num_chars = sum(1 for _ in f) + 1
    
    vocab_chars = tf.contrib.lookup.index_table_from_file(
      char_vocab_file, num_oov_buckets=1
    )
    
    char_ids = vocab_chars.lookup(chars)

    v = tf.get_variable('char_embeddings', [num_chars + 1, char_embedding_size], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(v, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=0.5, training=training)
    return char_embeddings

def get_char_representations(chars, nchars, char_vocab_file, mode='cnn', training=False, scope='None'):
  char_embedding_size = 50
  char_embeddings = get_char_embeddings(chars, char_vocab_file, char_embedding_size)

  filters = 50
  kernel_size = 3
  char_lstm_size = 25

  if mode == 'cnn':
    weights = tf.sequence_mask(nchars)
    char_embeddings = cnn_char_representations(char_embeddings, weights, filters, kernel_size) 
  elif mode == 'lstm':
    char_embeddings = lstm_char_representations(char_embeddings, nchars, char_lstm_size, char_embedding_size, scope=scope)
  return char_embeddings
