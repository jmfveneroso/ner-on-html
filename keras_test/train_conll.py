from util import Conll2003, Dataset, WordEmbeddings
from lstm_crf import LstmCrf


we       = WordEmbeddings('glove-50')
dev      = Conll2003(we, 'conll-2003/eng.train', max_sentence_len=50, max_token_len=20)
validate = Conll2003(we, 'conll-2003/eng.testa', max_sentence_len=50, max_token_len=20)
test     = Conll2003(we, 'conll-2003/eng.testb', max_sentence_len=50, max_token_len=20)

# print(dev.X2.shape)
# print(validate.X2.shape)
# print(test.X2.shape)

lstm_crf = LstmCrf(
  'lstm-crf-conll',
  model_type='lstm-crf', 
  num_labels=9,
  dev_dataset=dev,
  validate_dataset=validate,
  test_dataset=test
)
lstm_crf.create(we.matrix)
lstm_crf.fit(epochs=5)
