import tensorflow as tf
from official.nlp.bert.tokenization import FullTokenizer
from official.nlp.bert.squad_lib import (
    read_squad_examples, convert_examples_to_features, write_predictions,
    FeatureWriter )


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

  # logging.info('***** Running predictions *****')
  # logging.info('  Num orig examples = %d', len(eval_examples))
  # logging.info('  Num split examples = %d', len(eval_features))
  # logging.info('  Batch size = %d', predict_batch_size)

  # num_steps = int(dataset_size / predict_batch_size)
  # all_results = predict_squad_customized(strategy, input_meta_data, bert_config,
  #                                        eval_writer.filename, num_steps)

  # output_prediction_file = os.path.join(FLAGS.model_dir, 'predictions.json')
  # output_nbest_file = os.path.join(FLAGS.model_dir, 'nbest_predictions.json')
  # output_null_log_odds_file = os.path.join(FLAGS.model_dir, 'null_odds.json')

  # write_predictions(
  #     eval_examples,
  #     eval_features,
  #     all_results,
  #     FLAGS.n_best_size,
  #     FLAGS.max_answer_length,
  #     FLAGS.do_lower_case,
  #     output_prediction_file,
  #     output_nbest_file,
  #     output_null_log_odds_file,
  #     verbose=FLAGS.verbose_logging)

