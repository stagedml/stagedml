from stagedml.types import ( Glue, GlueTFR, BertCP, Optional, Any, List, Tuple,
                            Union )

from stagedml.core import ( json_read, readlines )

from stagedml.imports.sys import (environ, join, makedirs, mkdir, json_dumps,
                                  shuffle, RefPath, Manager, Build, Config,
                                  Hash, DRef, build_cattrs, build_outpath,
                                  build_path, mkdrv, match_only, mklens,
                                  promise, mkconfig, build_wrapper,
                                  match_latest, OrderedDict, set_trace)

from stagedml.imports.tf import ( TFRecordWriter, Example, Features, Feature,
                                 Int64List )

from stagedml.datasets.glue.download_glue_data import TASKS as GLUE_TASKS

from stagedml.datasets.glue.processors import ( get_processor, InputExample )

from keras_bert import Tokenizer, load_vocabulary

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

def make(b:Build)->None:
  c=build_cattrs(b)
  _=build_outpath(b)
  print(f"Processing {c.task_name}..")

  task_name=c.task_name
  data_dir=mklens(b).inputdir.syspath
  _=mklens(b).bert_vocab.syspath
  max_seq_length=mklens(b).max_seq_length.val

  processor = get_processor(task_name)

  train_valid_examples=processor.get_train_examples(data_dir)
  test_examples=processor.get_dev_examples(data_dir)
  shuffle(train_valid_examples)
  train_len=int(len(train_valid_examples)*c.train_valid_ratio)
  train_examples=train_valid_examples[:train_len]
  valid_examples=train_valid_examples[train_len:]
  label_ids = {label:i for i,label in enumerate(processor.get_labels())}

  vocab = load_vocabulary(mklens(b).bert_vocab.syspath)
  tokenizer = Tokenizer(vocab, cased=not mklens(b).lower_case.val)

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
    version = 9
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
      matcher=match_only(),
      realizer=build_wrapper(make)))

