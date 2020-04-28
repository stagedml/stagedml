from stagedml.imports import makedirs
from stagedml.stages.all import *







from pylightnix import RRef, rref2path, match_some, realizeMany
from stagedml.types import Dict, Union, Optional
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
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

def tensorboard_tensor_events(
    rref:RRef, subfolder:str, tag:str)->ScalarEvent:
  path=join(rref2path(rref),subfolder)
  event_acc=EventAccumulator(path, {
      'compressedHistograms': 10, 'images': 0,
      'scalars': 100, 'histograms': 1 })
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


def experiment_bs()->dict:
  result={}
  n=5
  for bs in [2,8,16,32,64]:
    def _new_config(cfg:dict):
      cfg['train_batch_size']=bs
      cfg['train_epoches']=5
      return mkconfig(cfg)
    result[bs]=realizeMany(instantiate(
      redefine(all_minibert_finetune_glue, new_config=_new_config,
                                           new_matcher=match_some(n)),
      num_instances=n))
  return result


