import tensorflow as tf

from official.nlp.bert.tokenization import FullTokenizer
from official.nlp.bert.squad_lib import (
    read_squad_examples, convert_examples_to_features, write_predictions,
    FeatureWriter )
from official.nlp.bert.input_pipeline import decode_record


def generate_tf_record_from_json_file(input_file_path:str,
                                      vocab_file_path:str,
                                      output_path:str,
                                      max_seq_length:int=384,
                                      do_lower_case:bool=True,
                                      max_query_length:int=64,
                                      doc_stride:int=128,
                                      version_2_with_negative=False)->int:
  """Generates and saves training data into a tf record file."""
  train_examples = read_squad_examples(
      input_file=input_file_path,
      is_training=True,
      version_2_with_negative=version_2_with_negative)
  tokenizer = FullTokenizer(vocab_file=vocab_file_path,
      do_lower_case=do_lower_case)
  train_writer = FeatureWriter(filename=output_path, is_training=True)
  number_of_examples = convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)

  train_writer.close()

  return number_of_examples




def predict_squad(input_file_path, output_file, vocab_file,
                  doc_stride, predict_batch_size, max_query_length,
                  max_seq_length, do_lower_case,
                  version_2_with_negative=False)->int:
  """Makes predictions for a squad dataset."""

  eval_examples = read_squad_examples(
      input_file=input_file_path,
      is_training=False,
      version_2_with_negative=version_2_with_negative)

  tokenizer = FullTokenizer(vocab_file=vocab_file,
      do_lower_case=do_lower_case)

  eval_writer = FeatureWriter(filename=output_file, is_training=False)

  eval_features = []

  def _append_feature(feature, is_padding):
    if not is_padding:
      eval_features.append(feature)
    eval_writer.process_feature(feature)

  # TPU requires a fixed batch size for all batches, therefore the number
  # of examples must be a multiple of the batch size, or else examples
  # will get dropped. So we pad with fake examples which are ignored
  # later on.
  number_of_examples = convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=False,
      output_fn=_append_feature,
      batch_size=predict_batch_size)
  eval_writer.close()

  return number_of_examples


def tf_record_dataset(input_file:str,
                      max_seq_length:int,
                      train_batch_size:int)->tf.data.TFRecordDataset:

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



