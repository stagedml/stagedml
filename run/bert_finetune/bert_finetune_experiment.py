from pylightnix import ( RRef, rref2path, match_some, realizeMany, match_latest,
    store_buildtime, store_buildelta )

from stagedml.imports import ( makedirs, MakeNdarray, join, environ, makedirs,
    defaultdict, getcwd, DataFrame, read_csv )
from stagedml.stages.all import *
from stagedml.types import ( Dict, Union, Optional, List )
from stagedml.core import ( protocol_rref_metric )
from stagedml.utils import ( tensorboard_tensors, tensorboard_scalars,
    tensorboard_tags, te2float )

import numpy as np

from altair import Chart
from altair_saver import save as altair_save


def altair_print(chart:Chart, png_filename:str, alt:str='', attrs:str='')->None:
  genimgdir=environ['REPOUT']
  repimgdir=environ.get('REPIMG',genimgdir)
  makedirs(genimgdir, exist_ok=True)
  altair_save(chart, join(genimgdir,png_filename))
  print("![%s](%s){%s}"%(alt, join(repimgdir,png_filename), attrs))


def experiment_bs(n:int=4, exclude=[])->Dict[int,List[RRef]]:
  result_bs={}
  for bs in [2,8,16,32,64]:
    def _new_config(cfg:dict):
      cfg['train_batch_size']=bs
      cfg['train_epoches']=5
      cfg['flags']=[f for f in cfg['flags'] if f not in exclude]
      return mkconfig(cfg)
    result_bs[bs]=realizeMany(instantiate(
      redefine(all_minibert_finetune_glue, new_config=_new_config,
                                           new_matcher=match_some(n)),
      num_instances=n))
  return result_bs

def experiment_lr(n:int=4, exclude=[])->Dict[str,List[RRef]]:
  result_lr={}
  for lr in [3e-4, 1e-4, 5e-5, 3e-5]:
    def _new_config(cfg:dict):
      cfg['train_batch_size']=8
      cfg['lr']=lr
      cfg['train_epoches']=5
      cfg['flags']=[f for f in cfg['flags'] if f not in exclude]
      return mkconfig(cfg)
    result_lr[str(lr)]=realizeMany(instantiate(
      redefine(all_minibert_finetune_glue, new_config=_new_config,
                                           new_matcher=match_some(n)),
      num_instances=n))
  return result_lr


def experiment_trainmethod()->Dict[str,RRef]:
  result={}
  for tm in ['fit','custom']:
    def _new_config(cfg:dict):
      cfg['train_epoches']=5
      cfg['train_method']=tm
      return mkconfig(cfg)
    result[tm]=realize(instantiate(
      redefine(all_minibert_finetune_glue,
        new_config=_new_config, new_matcher=match_latest())))
  return result


def experiment_allglue(n:int=1)->Dict[str,List[RRef]]:
  result_allglue={}
  for task_name in [t for t in glue_tasks() if t.upper() not in ['COLA']]:
    print(f"Fine-tuning {task_name}")
    batch_size={'MNLI-M':64,
                'MNLI-MM':64,
                'SNLI':64}.get(task_name.upper(),8)
    def _new_config(cfg:dict):
      cfg['train_batch_size']=batch_size
      cfg['train_epoches']=4
      return mkconfig(cfg)
    result_allglue[task_name]=realizeMany(instantiate(
      redefine(all_minibert_finetune_glue,
        new_config=_new_config, new_matcher=match_some(n)),
      task_name=task_name, num_instances=n))
  return result_allglue



if __name__== '__main__':
  print(experiment_allglue())
  print(experiment_bs())
  print(experiment_lr())

