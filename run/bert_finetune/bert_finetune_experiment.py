from stagedml.imports import makedirs
from stagedml.stages.all import *

from pylightnix import ( RRef, rref2path, match_some, realizeMany, match_latest,
    store_buildtime, store_buildelta )
from stagedml.types import Dict, Union, Optional
from stagedml.core import ( protocol_rref_metric )
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE )
from tensorflow.python.framework.tensor_util import MakeNdarray
from tensorboard.backend.event_processing.event_accumulator import (
    ScalarEvent, TensorEvent )
from tensorflow.python.framework.tensor_util import MakeNdarray
from typing import List


def tensorboard_tags(rref:RRef,subfolder:str='train')->Dict[str,Union[list,bool]]:
  path=join(rref2path(rref),subfolder)
  event_acc = EventAccumulator(path, {
      'compressedHistograms': 10,
      'images': 0,
      'tensors':10,
      'scalars': 100,
      'histograms': 1})
  event_acc.Reload()
  return event_acc.Tags()

def tensorboard_scalar_events(
    rref:RRef, subfolder:str, tag:str)->ScalarEvent:
  path=join(rref2path(rref),subfolder)
  event_acc=EventAccumulator(path, {
      'compressedHistograms': 10, 'images': 0,
      'scalars': 100, 'histograms': 1 })
  event_acc.Reload()
  return event_acc.Scalars(tag)

def tensorboard_tensor_events(rref:RRef, subfolder:str, tag:str)->ScalarEvent:
  path=join(rref2path(rref),subfolder)
  event_acc=EventAccumulator(path, {'scalars': 1000, 'tensors':1000 })
  event_acc.Reload()
  return event_acc.Tensors(tag)

def te2float(te:TensorEvent):
  """ Should be a bug in TF """
  return float(MakeNdarray(te.tensor_proto).tolist()) # SIC!



import altair as alt
from altair import Chart
import pandas as pd
from pandas import DataFrame
from stagedml.imports import ( join, environ, makedirs, defaultdict, getcwd )
from altair_saver import save as altair_save


def altair_print(chart:Chart, png_filename:str, alt:str='', attrs:str='')->None:
  genimgdir=environ['REPOUT']
  repimgdir=environ.get('REPIMG',genimgdir)
  makedirs(genimgdir, exist_ok=True)
  altair_save(chart, join(genimgdir,png_filename))
  print("![%s](%s){%s}"%(alt, join(repimgdir,png_filename), attrs))


def experiment_bs(n:int=1, exclude=[])->Dict[int,List[RRef]]:
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



