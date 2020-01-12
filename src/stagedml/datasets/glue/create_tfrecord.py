import json
import logging
import os
import random
import sys

from absl import app
from absl import flags
import tensorflow as tf

from official.nlp.bert import tokenization
from official.nlp.bert.classifier_data_lib import file_based_convert_examples_to_features
from stagedml.datasets.glue.processors import get_processor


def create_tfrecord_data(task_name, data_dir, vocab_path, output_dir,
                         max_seq_length=128, names=['train', 'test', 'dev'],
                         lower_case:bool=True):
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  meta_data = {
    "task_name": task_name,
    "max_seq_length": max_seq_length,
  }
  tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=lower_case)
  processor = get_processor(task_name)

  for name in names:
    output_path = os.path.join(output_dir, f'{name}.tfrecord')
    if name=='train':
      examples = processor.get_train_examples(data_dir)
    elif name=='test':
      examples = processor.get_test_examples(data_dir)
    elif name=='dev':
      examples = processor.get_dev_examples(data_dir)
    else:
      raise ValueError(f'Unknown task name {task_name}')
    labels = processor.get_labels()
    meta_data.update({f'{name}_data_size':len(examples)})
    file_based_convert_examples_to_features(examples, labels, max_seq_length, tokenizer, output_path)

  meta_data.update({"num_classes": len(processor.get_labels())})

  meta_data_path = os.path.join(output_dir, 'meta.json')
  with tf.io.gfile.GFile(meta_data_path, "w") as f:
    f.write(json.dumps(meta_data, indent=4) + "\n")




