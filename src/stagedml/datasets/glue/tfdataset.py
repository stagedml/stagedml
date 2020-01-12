import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from os import environ
from official.nlp.bert.input_pipeline import decode_record
from pylightnix import Config, config_ro

def dataset(data_path:str, config:Config):
  c=config_ro(config)
  max_seq_length = c.task_config['max_seq_length']
  d = tf.data.TFRecordDataset(data_path)
  name_to_features = {
    'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
    'input_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
    'segment_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
    'label_ids': tf.io.FixedLenFeature([], tf.int64),
    'is_real_example': tf.io.FixedLenFeature([], tf.int64),
  }
  d = d.map(lambda record: decode_record(record, name_to_features))

  def _select_data_from_record(record):
    x = {
      'input_word_ids': record['input_ids'],
      'input_mask': record['input_mask'],
      'input_type_ids': record['segment_ids']
    }
    y = record['label_ids']
    return (x, y)

  d = d.map(_select_data_from_record)
  return d

def dataset_train(path:str, config:Config):
  c = config_ro(config)
  d = dataset(path, config)
  d = d.shuffle(100)

  dtrain=d.take(c.train_data_size)
  dtrain=dtrain.repeat()
  dtrain=dtrain.batch(c.train_batch_size, drop_remainder=True)
  dtrain=dtrain.prefetch(1024)

  dvalid=d.skip(c.train_data_size)
  dvalid=dvalid.batch(c.train_batch_size, drop_remainder=False)
  return dtrain,dvalid

def dataset_eval(path:str, config:Config):
  c = config_ro(config)
  d = dataset(path, config)
  d = d.batch(c.eval_batch_size, drop_remainder=False)
  d = d.prefetch(1024)
  return d


