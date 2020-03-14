import numpy as np
# from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.python.framework.tensor_util import MakeNdarray

from typing import List

def _accl2float(av):
  lst=MakeNdarray(av.tensor_proto).tolist()
  return float(lst) # SIC!


def read_tensorflow_log(path:str, tag:str):
  event_acc = EventAccumulator(path, {
      'compressedHistograms': 10,
      'images': 0,
      'scalars': 100,
      'histograms': 1
  })
  event_acc.Reload()

  # Show all tags in the log file
  # print(event_acc.Tags())
  data = event_acc.Tensors(tag)
  steps = len(data)
  # print(steps)
  y = np.zeros([steps])

  for i in range(steps):
    y[i] = _accl2float(data[i])
  return y



from os import environ
from pylightnix import ( repl_realize, repl_buildargs, repl_cancel, instantiate,
    build_setoutpaths,repl_cancelBuild )
import stagedml.stages.transformer_wmt as twmt
import tensorflow as tf
from logging import getLogger
getLogger('tensorflow').setLevel('FATAL')
tf.autograph.set_verbosity(3, False)

def model_size(stage)->int:

  th=repl_realize(instantiate(stage))
  b=twmt.TransformerBuild(repl_buildargs(th))
  try:
    build_setoutpaths(b,1)
    twmt.build(b,0)
    msize=0
    for p in b.train_model.trainable_weights:
      sz=1
      for dim in p.shape:
        sz*=dim
      msize+=sz
    return msize
  finally:
    repl_cancelBuild(b,th)

