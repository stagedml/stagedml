import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from official.nlp.bert.input_pipeline import decode_record

def squad11_train_dataset(input_file:str, max_seq_length:int, train_batch_size:int):
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'start_positions': tf.io.FixedLenFeature([], tf.int64),
      'end_positions': tf.io.FixedLenFeature([], tf.int64),
  }

  d = tf.data.TFRecordDataset(input_file)
  d = d.map(lambda record: decode_record(record, name_to_features))

  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = (
      tf.data.experimental.AutoShardPolicy.OFF)
  d = d.with_options(options)

  def _select_data_from_record(record):
    x, y = {}, {}
    for name, tensor in record.items():
      if name in ('start_positions', 'end_positions'):
        y[name] = tensor
      elif name == 'input_ids':
        x['input_word_ids'] = tensor
      elif name == 'segment_ids':
        x['input_type_ids'] = tensor
      else:
        x[name] = tensor
    return (x, y)

  d = d.map(_select_data_from_record)
  d = d.shuffle(100)
  d = d.repeat()
  d = d.batch(train_batch_size, drop_remainder=True)
  d = d.prefetch(1024)
  return d



# if is_training:
#   dataset = dataset.shuffle(100)
#   dataset = dataset.repeat()

# dataset = dataset.batch(batch_size, drop_remainder=True)
# dataset = dataset.prefetch(1024)
# return dataset

# def dataset_train(path:str, config:Config):
#   c = config_ro(config)
#   d = dataset(path, config)
#   d = d.shuffle(100)

#   dtrain=d.take(c.train_data_size)
#   dtrain=dtrain.repeat()
#   dtrain=dtrain.batch(c.train_batch_size, drop_remainder=True)
#   dtrain=dtrain.prefetch(1024)

#   dvalid=d.skip(c.train_data_size)
#   dvalid=dvalid.batch(c.train_batch_size, drop_remainder=False)
#   return dtrain,dvalid

# def dataset_eval(path:str, config:Config):
#   c = config_ro(config)
#   d = dataset(path, config)
#   d = d.batch(c.eval_batch_size, drop_remainder=False)
#   d = d.prefetch(1024)
#   return d



