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

    # plt.plot(x, y[:,0], label='bleu_cased')
    # plt.plot(x, y[:,1], label='bleu_uncased')

    # plt.xlabel("Steps")
    # plt.title("Training Progress")
    # plt.legend(loc='upper right', frameon=True)
    # # plt.savefig('fig1.png')
    # plt.show()



# O="_pylightnix/experiments/nl2bash/10000/eval"
# plot_tensorflow_log(O)




# import tensorflow as tf

# from tensorflow.compat.v1.train import summary_iterator

# def method2(p):
#   for e in tf.compat.v1.train.summary_iterator(p):
#       for v in e.summary.value:
#         if v.tag == 'bleu_cased':
#           print(v.simple_value)


# from pylightnix import *
# rref=path2rref(O)
# from stagedm import *
