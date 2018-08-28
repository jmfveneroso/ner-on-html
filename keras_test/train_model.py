from util import Dataset, WordEmbeddings
from lstm_crf import LstmCrf

we       = WordEmbeddings('glove-50')
dev      = Dataset(we, 'dataset/dev.txt', max_sentence_len=50, max_token_len=20)
validate = Dataset(we, 'dataset/validate.txt', max_sentence_len=50, max_token_len=20)
test     = Dataset(we, 'dataset/test.txt', max_sentence_len=50, max_token_len=20)

lstm_crf = LstmCrf(
  'lstm-crf-cnn',
  model_type='lstm-crf-cnn', 
  dev_dataset=dev,
  validate_dataset=validate,
  test_dataset=test
)
lstm_crf.create(we.matrix)
lstm_crf.fit(epochs=10)

print('\n\n')

lstm_crf.print_names()
