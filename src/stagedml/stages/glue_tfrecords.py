from os import environ, makedirs
from os.path import join
from typing import Optional,Any,List,Tuple,Union

from pylightnix import ( Manager, Build, Config, Hash, DRef, build_cattrs,
    build_outpath, build_path, mkdrv, only )

from stagedml.utils.files import json_read
from stagedml.utils.tf import ( KerasBuild, protocol_add, memlimit, keras_realizer )
# from stagedml.utils.instantiate import Options, instantiate
from stagedml.utils.refs import Glue,GlueTFR,BertCP
from stagedml.datasets.glue.create_tfrecord import create_tfrecord_data
from stagedml.datasets.glue.download_glue_data import TASKS as GLUE_TASKS

def glue_tasks():
  return [t for t in GLUE_TASKS if t != 'diagnostic' and t != 'STS' and t != 'MNLI'] + ['MNLI-m','MNLI-mm']


def config(dataset_name:str, refbert:BertCP, refdataset:Glue)->Config:
  version=5
  name='tfrecord'
  if dataset_name == 'glue':
    tasks=glue_tasks()
  else:
    raise ValueError('Unsupported dataset name {dataset_name}')
  bert_refpath = [refbert, 'uncased_L-12_H-768_A-12']
  bert_ckpt_refpath = bert_refpath+['bert_model.ckpt']
  bert_config = bert_refpath+['bert_config.json']
  bert_vocab_refpath = bert_refpath+['vocab.txt']
  max_seq_length = 128
  return Config(locals())


def process(b:KerasBuild)->None:
  c=build_cattrs(b)
  o=build_outpath(b)
  if c.dataset_name=='glue':
    def _run(task_name:str, task_dir:str):
      print(f'Processing {task_name}..')
      task_out_dir=join(o,task_name)
      makedirs(task_out_dir)
      create_tfrecord_data(task_name=task_name,
                           data_dir=build_path(b, [c.refdataset, task_dir]),
                           vocab_path=build_path(b, c.bert_vocab_refpath),
                           output_dir=task_out_dir,
                           max_seq_length=c.max_seq_length)
    for task_name in c.tasks:
      if 'SST' in task_name:
        _run(task_name,'SST-2')
      elif 'MNLI' in task_name:
        _run(task_name, 'MNLI')
      else:
        _run(task_name,task_name)
  else:
    raise ValueError('Unsupported dataset name {c.dataset_name}')
  protocol_add(b, 'process')


def glue_tfrecords(m:Manager, dataset_name:str, refbert:BertCP, refdataset:Glue)->GlueTFR:
  c=config(dataset_name, refbert, refdataset)
  def _instantiate():
    return config(dataset_name, refbert, refdataset)
  return GlueTFR(mkdrv(m, _instantiate, only(), keras_realizer(process)))


