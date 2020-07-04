from os import environ, makedirs
from os.path import join
from typing import Optional,Any,List,Tuple,Union

from pylightnix import ( RefPath, Manager, Build, Config, Hash, DRef,
    build_cattrs, build_outpath, build_path, mkdrv, match_only, mklens,
    promise, mkconfig, build_wrapper )

from stagedml.types import Glue,GlueTFR,BertCP
from stagedml.utils.files import ( json_read )
from stagedml.imports.sys import ( join, mkdir, json_dumps, shuffle )
from stagedml.imports.tf import ( FullTokenizer,
    file_based_convert_examples_to_features )
from stagedml.datasets.glue.download_glue_data import TASKS as GLUE_TASKS

from stagedml.datasets.glue.processors import ( get_processor, InputExample )

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

def process(b:Build)->None:
  c=build_cattrs(b)
  o=build_outpath(b)
  print(f"Processing {c.task_name}..")

  task_name=c.task_name
  data_dir=mklens(b).inputdir.syspath
  vocab_path=mklens(b).bert_vocab.syspath

  tokenizer = FullTokenizer(vocab_file=vocab_path, do_lower_case=c.lower_case)
  processor = get_processor(task_name)

  train_valid_examples=processor.get_train_examples(data_dir)
  test_examples=processor.get_dev_examples(data_dir)
  shuffle(train_valid_examples)
  train_len=int(len(train_valid_examples)*c.train_valid_ratio)
  train_examples=train_valid_examples[:train_len]
  valid_examples=train_valid_examples[train_len:]

  # Train
  file_based_convert_examples_to_features(
    train_examples,
    processor.get_labels(),
    c.max_seq_length,
    tokenizer,
    mklens(b).outputs.train.syspath)

  # Valid
  file_based_convert_examples_to_features(
    valid_examples,
    processor.get_labels(),
    c.max_seq_length,
    tokenizer,
    mklens(b).outputs.valid.syspath)

  # Test
  file_based_convert_examples_to_features(
    test_examples,
    processor.get_labels(),
    c.max_seq_length,
    tokenizer,
    mklens(b).outputs.test.syspath)

def glue_tfrecords(m:Manager,
                   task_name:str,
                   bert_vocab:RefPath,
                   lower_case:bool,
                   refdataset:Glue)->GlueTFR:

  assert task_name in glue_tasks(), \
      f"Unsupported task '{task_name}'. Expected one of {glue_tasks()}"

  def _config():
    version=6
    name='tfrecord-'+task_name.lower()
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
      realizer=build_wrapper(process)))


