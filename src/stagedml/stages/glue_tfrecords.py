from os import environ, makedirs
from os.path import join
from typing import Optional,Any,List,Tuple,Union

from pylightnix import ( RefPath, Manager, Build, Config, Hash, DRef,
    build_cattrs, build_outpath, build_path, mkdrv, match_only, mklens,
    promise, mkconfig, build_wrapper )

from stagedml.utils import ( json_read, memlimit )
from stagedml.types import Glue,GlueTFR,BertCP
from stagedml.datasets.glue.create_tfrecord import create_tfrecord_data
from stagedml.datasets.glue.download_glue_data import TASKS as GLUE_TASKS

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
  task_out_dir=mklens(b).outputs.dir.syspath
  makedirs(task_out_dir)
  create_tfrecord_data(task_name=c.task_name,
                       data_dir=mklens(b).inputdir.syspath,
                       vocab_path=mklens(b).bert_vocab.syspath,
                       output_dir=task_out_dir,
                       max_seq_length=c.max_seq_length)


def glue_tfrecords(m:Manager,
                   task_name:str,
                   bert_vocab:RefPath,
                   refdataset:Glue)->GlueTFR:

  assert task_name in glue_tasks(), \
      f"Unsupported task '{task_name}'. Expected one of {glue_tasks()}"

  def _config():
    version=6
    name='tfrecord-'+task_name.lower()
    nonlocal bert_vocab
    inputdir = [refdataset,_glue_task_src(task_name)]
    outputs={'dir':[promise,task_name],
             'train':[promise,task_name,'train.tfrecord'],
             'dev':[promise,task_name,'dev.tfrecord'],
             'meta':[promise,task_name,'meta.json']}
    max_seq_length = 128
    return locals()

  return GlueTFR(
    mkdrv(m,
      config=mkconfig(_config()),
      matcher=match_only(),
      realizer=build_wrapper(process)))


