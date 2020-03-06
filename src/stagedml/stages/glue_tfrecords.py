from os import environ, makedirs
from os.path import join
from typing import Optional,Any,List,Tuple,Union

from pylightnix import ( Manager, Build, Config, Hash, DRef, build_cattrs,
    build_outpath, build_path, mkdrv, match_only, mklens, promise, mkconfig )

from stagedml.utils import ( json_read, memlimit )
from stagedml.types import Glue,GlueTFR,BertCP
from stagedml.datasets.glue.create_tfrecord import create_tfrecord_data
from stagedml.datasets.glue.download_glue_data import TASKS as GLUE_TASKS
from stagedml.core import ( ProtocolBuild, protocol_add, protocolled )

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

def config(task_name:str, refbert:BertCP, refdataset:Glue)->Config:
  assert task_name in glue_tasks(), \
      f"Unsupported task '{task_name}'. Expected one of {glue_tasks()}"
  version=6
  name='tfrecord-'+task_name.lower()
  bert_vocab = mklens(refbert).bert_vocab.refpath
  inputdir = [refdataset,_glue_task_src(task_name)]
  outputs={'dir':[promise,task_name],
           'train':[promise,task_name,'train.tfrecord'],
           'dev':[promise,task_name,'dev.tfrecord'],
           'meta':[promise,task_name,'meta.json']}
  max_seq_length = 128
  return mkconfig(locals())


def process(b:ProtocolBuild)->None:
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
  protocol_add(b, 'process')


def glue_tfrecords(m:Manager, task_name:str, refbert:BertCP, refdataset:Glue)->GlueTFR:
  return GlueTFR(
    mkdrv(m,
      config=config(task_name, refbert, refdataset),
      matcher=match_only(),
      realizer=protocolled(process)))


