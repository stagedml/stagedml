import csv
import os
import sys

import tensorflow as tf

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if isinstance(text, str):
    return text
  elif isinstance(text, bytes):
    return text.decode("utf-8", "ignore")
  else:
    raise ValueError("Unsupported string type: %s" % (type(text)))

max_int = sys.maxsize
csv.field_size_limit(max_int)

class InputExample(object):
  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class DataProcessor(object):
  def get_train_examples(self, data_dir):
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    raise NotImplementedError()

  def get_labels(self):
    raise NotImplementedError()

  @staticmethod
  def get_processor_name():
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, has_header=True, delimiter="\t", quotechar=None):
    with tf.io.gfile.GFile(input_file, 'r') as f:
      reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
      header = None
      if has_header:
        header = next(reader)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class ImdbProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(data_dir, "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(data_dir, "test")

  def get_labels(self):
    return ["neg", "pos"]

  @staticmethod
  def get_processor_name():
    return "IMDB"

  def _create_examples(self, data_dir, set_type):
    data_dir = os.path.join(data_dir, set_type)
    examples = []
    for label in ["neg", "pos"]:
      cur_dir = os.path.join(data_dir, label)
      for idx, filename in enumerate(tf.io.gfile.listdir(cur_dir)):
        if not filename.endswith("txt"):
          continue
        path = os.path.join(cur_dir, filename)
        with tf.io.gfile.GFile(path) as f:
          text_a = f.read().strip().replace("<br />", " ")
        text_b = None
        guid = "{}_{}_{}".format(set_type, label, idx)
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class ColaProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv"), has_header=False), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv"), has_header=False), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv"), has_header=False), "test")

  def get_labels(self):
    return ['0', '1']

  @staticmethod
  def get_processor_name():
    return "CoLa"

  def _create_examples(self, lines, set_type):
    examples = []
    for idx, line in enumerate(lines):
      guid = "{}-{}".format(set_type, convert_to_unicode(line[0]))
      text_a = convert_to_unicode(line[-1])
      if set_type == "test":
        label = "0"
      else:
        label = convert_to_unicode(line[1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

class Sst2Processor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    return ['0', '1']

  @staticmethod
  def get_processor_name():
    return "SST-2"

  def _create_examples(self, lines, set_type):
    examples = []
    header = lines[0]
    for idx, line in enumerate(lines[1:]):
      guid = "{}-{}".format(set_type, idx)
      if set_type == "test":
        text_a = convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = convert_to_unicode(line[0])
        label = convert_to_unicode(line[1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

class MrpcProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    return ['0', '1']

  @staticmethod
  def get_processor_name():
    return "MRPC"

  def _create_examples(self, lines, set_type):
    examples = []
    header = lines[0]
    for idx, line in enumerate(lines[1:]):
      guid = "{}-{}".format(set_type, idx)
      text_a = convert_to_unicode(line[-2])
      text_b = convert_to_unicode(line[-1])
      if set_type == "test":
        label = "0"
      else:
        label = convert_to_unicode(line[0])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class QqpProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    return ['0', '1']

  @staticmethod
  def get_processor_name():
    return "QQP"

  def _create_examples(self, lines, set_type):
    examples = []
    header = lines[0]
    for idx, line in enumerate(lines[1:]):
      if len(line) != len(header):
        continue
      guid = "{}-{}".format(set_type, idx)
      if set_type == "test":
        text_a = convert_to_unicode(line[-2])
        text_b = convert_to_unicode(line[-1])
        label = "0"
      else:
        text_a = convert_to_unicode(line[-3])
        text_b = convert_to_unicode(line[-2])
        label = convert_to_unicode(line[-1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class MnliMatchedProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    return ["contradiction", "entailment", "neutral"]

  @staticmethod
  def get_processor_name():
    return "MNLI-m"

  def _create_examples(self, lines, set_type):
    examples = []
    header = lines[0]
    for idx, line in enumerate(lines[1:]):
      guid = "{}-{}".format(set_type, idx)
      text_a = convert_to_unicode(line[8])
      text_b = convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = convert_to_unicode(line[-1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class MnliMismatchedProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test")

  def get_labels(self):
    return ["contradiction", "entailment", "neutral"]

  @staticmethod
  def get_processor_name():
    return "MNLI-mm"

  def _create_examples(self, lines, set_type):
    examples = []
    header = lines[0]
    for idx, line in enumerate(lines[1:]):
      guid = "{}-{}".format(set_type, idx)
      text_a = convert_to_unicode(line[8])
      text_b = convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = convert_to_unicode(line[-1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class SnliProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    return ["contradiction", "entailment", "neutral"]

  @staticmethod
  def get_processor_name():
    return "SNLI"

  def _create_examples(self, lines, set_type):
    examples = []
    header = lines[0]
    for idx, line in enumerate(lines[1:]):
      b_idx = len(line) - 1
      while line[b_idx] in self.get_labels():
        b_idx -= 1
      a_idx = b_idx - 1
      guid = "{}-{}".format(set_type, idx)
      text_a = convert_to_unicode(line[a_idx])
      text_b = convert_to_unicode(line[b_idx])
      if set_type == "test":
        label = "contradiction"
      else:
        label = convert_to_unicode(line[-1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class QnliProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    return ['entailment', 'not_entailment']

  @staticmethod
  def get_processor_name():
    return "QNLI"

  def _create_examples(self, lines, set_type):
    examples = []
    header = lines[0]
    for idx, line in enumerate(lines[1:]):
      guid = "{}-{}".format(set_type, idx)
      text_a = convert_to_unicode(line[1])
      text_b = convert_to_unicode(line[2])
      if set_type == "test":
        label = "entailment"
      else:
        label = convert_to_unicode(line[-1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class RteProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    return ['entailment', 'not_entailment']

  @staticmethod
  def get_processor_name():
    return "RTE"

  def _create_examples(self, lines, set_type):
    examples = []
    header = lines[0]
    for idx, line in enumerate(lines[1:]):
      guid = "{}-{}".format(set_type, idx)
      text_a = convert_to_unicode(line[1])
      text_b = convert_to_unicode(line[2])
      if set_type == "test":
        label = "entailment"
      else:
        label = convert_to_unicode(line[-1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class WnliProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    return ['0', '1']

  @staticmethod
  def get_processor_name():
    return "WNLI"

  def _create_examples(self, lines, set_type):
    examples = []
    header = lines[0]
    for idx, line in enumerate(lines[1:]):
      guid = "{}-{}".format(set_type, idx)
      text_a = convert_to_unicode(line[1])
      text_b = convert_to_unicode(line[2])
      if set_type == "test":
        label = "0"
      else:
        label = convert_to_unicode(line[-1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

def get_processor(task_name):
  processors = {
      'imdb': ImdbProcessor,
      'cola': ColaProcessor,
      'sst-2': Sst2Processor,
      'mrpc': MrpcProcessor,
      'qqp': QqpProcessor,
      'mnli-m': MnliMatchedProcessor,
      'mnli-mm': MnliMismatchedProcessor,
      'snli': SnliProcessor,
      'qnli': QnliProcessor,
      'rte': RteProcessor,
      'wnli': WnliProcessor,
  }
  if task_name.lower() in processors.keys():
    return processors[task_name.lower()]()
  else:
    raise ValueError("task_name is '{}' but it should be in {}".format(task_name, list(processors.keys())))

