from stagedml.imports.sys import (environ, join, makedirs, mkdir, json_dumps,
                                  shuffle, RefPath, Manager, Build, Config,
                                  Hash, DRef, build_cattrs, build_outpath,
                                  build_path, mkdrv, match_only, mklens,
                                  promise, mkconfig, build_wrapper,
                                  match_latest, OrderedDict)

from stagedml.types import (Glue,GlueTFR,BertCP,Optional,Any,List,Tuple,Union)

from stagedml.core import (json_read)

from stagedml.imports.tf import (TFRecordWriter, Example, Features, Feature,
                                 Int64List)

# from official.utils.misc.keras_utils import set_session_config
# from official.nlp.bert.tokenization import FullTokenizer
# from official.nlp.data.classifier_data_lib import \
#     file_based_convert_examples_to_features, convert_single_example

from stagedml.datasets.glue.download_glue_data import TASKS as GLUE_TASKS
from stagedml.datasets.glue.processors import ( get_processor, InputExample )

from keras_bert import Tokenizer

from ipdb import set_trace

def glue_tasks()->List[str]:
  tasks=[]
  for t in GLUE_TASKS:
    if t=='STS' or t=='diagnostic':
      pass
    elif t=='MNLI':
      tasks.extend(['MNLI-m','MNLI-mm'])
    elif t=='SST':
      tasks.append('SST-2')
    else:
      tasks.append(t)
  return tasks

def _glue_task_src(tn:str)->str:
  return 'MNLI' if 'MNLI' in tn else tn  # 'MNLI-m' and 'MNLI-mm' are special


# from official.nlp.bert.tokenization import FullTokenizer
# from official.nlp.data.classifier_data_lib import \
#     file_based_convert_examples_to_features, convert_single_example

# def file_based_convert_examples_to_features(examples, label_list,
#                                             max_seq_length, tokenizer,
#                                             output_file):
#   """Convert a set of `InputExample`s to a TFRecord file."""

#   writer = tf.io.TFRecordWriter(output_file)

#   for (ex_index, example) in enumerate(examples):
#     if ex_index % 10000 == 0:
#       logging.info("Writing example %d of %d", ex_index, len(examples))

#   tokenizer.tokenize(first='unaffable', second='钢')

#     feature = convert_single_example(ex_index, example, label_list,
#                                      max_seq_length, tokenizer)

#     def create_int_feature(values):
#       f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
#       return f

#     features = collections.OrderedDict()
#     features["input_ids"] = create_int_feature(feature.input_ids)
#     features["input_mask"] = create_int_feature(feature.input_mask)
#     features["segment_ids"] = create_int_feature(feature.segment_ids)
#     features["label_ids"] = create_int_feature([feature.label_id])
#     features["is_real_example"] = create_int_feature(
#         [int(feature.is_real_example)])

#     tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#     writer.write(tf_example.SerializeToString())

#   writer.close()




def make(b:Build)->None:
  c=build_cattrs(b)
  o=build_outpath(b)
  print(f"Processing {c.task_name}..")

  task_name=c.task_name
  data_dir=mklens(b).inputdir.syspath
  vocab_path=mklens(b).bert_vocab.syspath
  max_seq_length=mklens(b).max_seq_length.val

  # tokenizer = FullTokenizer(vocab_file=vocab_path, do_lower_case=c.lower_case)
  processor = get_processor(task_name)

  train_valid_examples=processor.get_train_examples(data_dir)
  test_examples=processor.get_dev_examples(data_dir)
  shuffle(train_valid_examples)
  train_len=int(len(train_valid_examples)*c.train_valid_ratio)
  train_examples=train_valid_examples[:train_len]
  valid_examples=train_valid_examples[train_len:]
  label_ids = {label:i for i,label in enumerate(processor.get_labels())}

  with open(mklens(b).bert_vocab.syspath) as f:
    vocab = {text:val for val,text in enumerate(f.read().split())}

  tokenizer = Tokenizer(vocab, cased=mklens(b).lower_case.val)

  def _tokenize(examples:List[InputExample], outfile):
    def _ints(values):
      return Feature(int64_list=Int64List(value=list(values)))
    with TFRecordWriter(outfile) as writer:
      for (i,e) in enumerate(examples):
        input_ids,segment_ids = tokenizer.encode(e.text_a, e.text_b,
                                                 max_seq_length)
        label_id = label_ids[e.label]

        features = OrderedDict()
        features["input_ids"] = _ints(input_ids)
        features["segment_ids"] = _ints(segment_ids)
        features["label_ids"] = _ints([label_id])
        te = Example(features=Features(feature=features))

        writer.write(te.SerializeToString())

  _tokenize(train_examples, mklens(b).outputs.train.syspath)
  _tokenize(valid_examples, mklens(b).outputs.valid.syspath)
  _tokenize(test_examples, mklens(b).outputs.test.syspath)

def glue_tfrecords(m:Manager,
                   task_name:str,
                   bert_vocab:RefPath,
                   lower_case:bool,
                   refdataset:Glue)->GlueTFR:

  assert task_name in glue_tasks(), \
      f"Unsupported task '{task_name}'. Expected one of {glue_tasks()}"

  def _config():
    version = 7
    name = 'tfrecord-'+task_name.lower()
    nonlocal bert_vocab
    nonlocal lower_case
    inputdir = [refdataset,_glue_task_src(task_name)]
    outputs={'train':[promise,'train.tfrecord'],
             'valid':[promise,'valid.tfrecord'],
             'test':[promise,'test.tfrecord'],
             'eval':None}
    max_seq_length = 128
    train_valid_ratio = 0.95
    num_classes = len(get_processor(task_name).get_labels())
    return locals()

  return GlueTFR(
    mkdrv(m,
      config=mkconfig(_config()),
      matcher=match_latest(), # FIXME: should be match_only
      realizer=build_wrapper(make)))


