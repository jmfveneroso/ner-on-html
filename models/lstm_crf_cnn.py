import numpy as np 

from six.moves import reduce
import tensorflow as tf

def model_fn(features, labels, mode, params):
  # For serving features are a bit different
  if isinstance(features, dict):
    features = ((features['words'], features['nwords']),
          (features['chars'], features['nchars']))  
    
  # Read vocabs and inputs
  dropout = params['dropout']
  (words, nwords), (chars, nchars) = features
  training = (mode == tf.estimator.ModeKeys.TRAIN)
  vocab_words = tf.contrib.lookup.index_table_from_file(
    params['words'], num_oov_buckets=params['num_oov_buckets'])
  vocab_chars = tf.contrib.lookup.index_table_from_file(
    params['chars'], num_oov_buckets=params['num_oov_buckets'])
  with Path(params['tags']).open() as f:
    indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
    num_tags = len(indices) + 1
  with Path(params['chars']).open() as f:
    num_chars = sum(1 for _ in f) + params['num_oov_buckets']

  # Char Embeddings
  char_ids = vocab_chars.lookup(chars)
  variable = tf.get_variable(
    'chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
  char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
  char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                    training=training)

  # Char 1d convolution
  weights = tf.sequence_mask(nchars)
  char_embeddings = masked_conv1d_and_max(
    char_embeddings, weights, params['filters'], params['kernel_size'])

  # Word Embeddings
  word_ids = vocab_words.lookup(words)
  glove = np.load(params['glove'])['embeddings']  # np.array
  variable = np.vstack([glove, [[0.] * params['dim']]])
  variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
  word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

  # Concatenate Word and Char Embeddings
  embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
  embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

  # LSTM
  t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
  lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
  lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
  lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
  output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
  output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
  output = tf.concat([output_fw, output_bw], axis=-1)
  output = tf.transpose(output, perm=[1, 0, 2])
  output = tf.layers.dropout(output, rate=dropout, training=training)

  # CRF
  logits = tf.layers.dense(output, num_tags)
  crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
  pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Predictions
    reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
      params['tags'])
    pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
    predictions = {
      'pred_ids': pred_ids,
      'tags': pred_strings
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  else:
    # Loss
    vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
    tags = vocab_tags.lookup(labels)
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
      logits, tags, nwords, crf_params)
    loss = tf.reduce_mean(-log_likelihood)

    # Metrics
    weights = tf.sequence_mask(nwords)
    metrics = {
      'acc': tf.metrics.accuracy(tags, pred_ids, weights),
      'precision': precision(tags, pred_ids, num_tags, indices, weights),
      'recall': recall(tags, pred_ids, num_tags, indices, weights),
      'f1': f1(tags, pred_ids, num_tags, indices, weights),
    }
    for metric_name, op in metrics.items():
      tf.summary.scalar(metric_name, op[1])

    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)

    elif mode == tf.estimator.ModeKeys.TRAIN:
      train_op = tf.train.AdamOptimizer().minimize(
        loss, global_step=tf.train.get_or_create_global_step())
      return tf.estimator.EstimatorSpec(
        mode, loss=loss, train_op=train_op)
