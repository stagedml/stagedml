""" Utility functions which require TensorFlow as a dependency """
import json
import numpy as np
import tensorflow as tf
assert tf.version.VERSION.startswith('2.1') or \
       tf.version.VERSION.startswith('2.2'), \
       (f"sbtagedml requires TensorFlow version '2.1.*' or '2.2.*', "
        f"not '{tf.version.VERSION}'")

from pylightnix import ( Closure, Path, Build, Hash, DRef, assert_valid_rref,
    assert_serializable, PYLIGHTNIX_TMP, Realizer, build_outpath, mkbuild, RRef,
    rref2path, readjson, json_dumps, store_rrefs, dirhash, Context,
    build_wrapper_, BuildArgs, repl_realize, repl_continue, repl_build, isrref )

from stagedml.imports.tf import ( TensorBoard, list_variables, History, Dataset,
    MakeNdarray )
from stagedml.imports.sys import ( Popen, join, remove, listdir, re_search, md5,
    os_run, Popen, default_timer )

from stagedml.types import ( Union, List, Any, Optional, Tuple, Callable,
    TypeVar, Dict )

FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]

from tensorflow.python.ops import summary_ops_v2

#  _   _ _   _ _
# | | | | |_(_) |___
# | | | | __| | / __|
# | |_| | |_| | \__ \
#  \___/ \__|_|_|___/

def memlimit(mem_gb:float)->None:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if tf.config.experimental.get_virtual_device_configuration(gpus[0]) is None:
    tf.config.experimental.set_virtual_device_configuration(
       gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_gb*1024)])

def ndhashl(arrays:List[np.array])->str:
  e=md5()
  for l in arrays:
    e.update(l)
  return e.hexdigest()

def ndhash(a:np.array)->str:
  return md5(a.tobytes()).hexdigest()

def thash(t:tf.Tensor)->str:
  return ndhash(t.numpy())

def nparams(x):
  res=0
  for x in x.weights:
      sz=1
      for d in x.shape:
          sz*=d if d is not None else 1
      res+=sz
  return res

def dpurge(dir, pattern, debug=True):
  for f in listdir(dir):
    if re_search(pattern, f):
      if debug:
        print('Removing', f, 'from', dir)
      remove(join(dir, f))

def runtensorboard(path:str, kill_existing:bool=True)->int:
  if kill_existing:
    os_run(('ps fax | grep -v grep | '
            'grep tensorboard | awk "{print \$1}" | '
            'xargs -r kill'), shell=True)
  with open(join(PYLIGHTNIX_TMP,"tensorboard.log"),"w") as f:
    pid = Popen(["tensorboard", "--host", "0.0.0.0", "--logdir", path],
                stdout=f, stderr=f).pid
  return pid

def runtb(arg:Union[RRef,Build,str])->None:
  if isrref(arg):
    path=rref2path(RRef(arg))
  elif isinstance(arg,str):
    path=Path(arg)
  elif isinstance(arg,Build):
    path=build_outpath(arg)
  else:
    assert False, "Value of unsupported type: '{arg}'"
  pid=runtensorboard(path)
  print('Tensorboard is running at', path, 'pid', pid)

def modelhash(m:tf.keras.Model)->Hash:
  return Hash(ndhashl(m.get_weights()))


def print_model_checkpoint_diff(m:tf.keras.Model, cprefix:str, tmpdir:Path)->None:
  l1=sorted([w.name.split(':')[0] for w in m.weights])
  l2=sorted([str(x[0]) for x in list_variables(cprefix)])
  # print('Model variables:\n','\n'.join())
  # print('Checkpoint variables:\n','\n'.join())
  def _savelist(l,fname):
    fname=join(tmpdir,fname)
    with open(fname,'w') as f:
      f.write('\n'.join(l))
    return fname
  Popen(['diff', '-y', '--color',
      _savelist(l1,'vars_model.txt'),
      _savelist(l2,'vars_checkpoint.txt')],
          shell=False, cwd='/').wait()
  print() # EOL


class TensorBoardFixed(TensorBoard):
  """ TensorBoard callback with a patch wich fixes training steps counter """
  def __init__(self, init_steps, *args, wall_clock_init:float=0.0, **kwargs):
    self.init_steps=init_steps
    self.wall_clock_init:float=wall_clock_init
    self.wall_clock_base:float=0.0
    self.wall_clock_last:float=0.0
    super().__init__(*args,**kwargs)

  def _init_batch_steps(self):
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import variables
    init_steps=self.init_steps
    if ops.executing_eagerly_outside_functions():
      self._total_batches_seen = {
          self._train_run_name: variables.Variable(init_steps, dtype='int64'),
          self._validation_run_name: variables.Variable(init_steps, dtype='int64')
      }
    else:
      self._total_batches_seen = {
          self._train_run_name: init_steps,
          self._validation_run_name: init_steps
      }

  def on_train_begin(self, logs=None):
    super().on_train_begin(logs=logs)
    self.wall_clock_base=default_timer()

  def on_train_batch_end(self, batch, logs=None):
    writer=self._get_writer(self._train_run_name)
    value=self.wall_clock_init+(default_timer()-self.wall_clock_base)
    self.wall_clock_last=value
    assert logs is not None
    logs.update({'wallclock':value})
    super().on_train_batch_end(batch, logs)

  def on_epoch_end(self, epoch, logs):
    writer=self._get_writer(self._train_run_name)
    value=self.wall_clock_init+(default_timer()-self.wall_clock_base)
    self.wall_clock_last=value
    assert logs is not None
    logs.update({'wallclock':value})
    super().on_epoch_end(epoch, logs)



class FBetaScore(tf.keras.metrics.Metric):
  def __init__(self,
               num_classes: FloatTensorLike,
               average: Optional[str] = None,
               beta: FloatTensorLike = 1.0,
               threshold: Optional[FloatTensorLike] = None,
               name: str = "fbeta_score",
               dtype: AcceptableDTypes = None,
               **kwargs):
    super().__init__(name=name, dtype=dtype)

    if average not in (None, "micro", "macro", "weighted"):
      raise ValueError(
        "Unknown average type. Acceptable values "
        "are: [None, micro, macro, weighted]")

    if not isinstance(beta, float):
      raise TypeError("The value of beta should be a python float")

    if beta <= 0.0:
      raise ValueError("beta value should be greater than zero")

    if threshold is not None:
      if not isinstance(threshold, float):
        raise TypeError("The value of threshold should be a python float")
      if threshold > 1.0 or threshold <= 0.0:
        raise ValueError("threshold should be between 0 and 1")

    self.num_classes = num_classes
    self.average = average
    self.beta = beta
    self.threshold = threshold
    self.axis = None
    self.init_shape = []

    if self.average != "micro":
      self.axis = 0
      self.init_shape = [self.num_classes]

    def _zero_wt_init(name):
      return self.add_weight(
          name, shape=self.init_shape, initializer="zeros", dtype=self.dtype)

    self.true_positives = _zero_wt_init("true_positives")
    self.false_positives = _zero_wt_init("false_positives")
    self.false_negatives = _zero_wt_init("false_negatives")
    self.weights_intermediate = _zero_wt_init("weights_intermediate")

  # TODO: Add sample_weight support, currently it is
  # ignored during calculations.
  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.threshold is None:
      threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
      # make sure [0, 0, 0] doesn't become [1, 1, 1]
      # Use abs(x) > eps, instead of x != 0 to check for zero
      y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
    else:
      y_pred = y_pred > self.threshold

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    def _count_non_zero(val):
      non_zeros = tf.math.count_nonzero(val, axis=self.axis)
      return tf.cast(non_zeros, self.dtype)

    self.true_positives.assign_add(_count_non_zero(y_pred * y_true))
    self.false_positives.assign_add(_count_non_zero(y_pred * (y_true - 1)))
    self.false_negatives.assign_add(_count_non_zero((y_pred - 1) * y_true))
    self.weights_intermediate.assign_add(_count_non_zero(y_true))

  def result(self):
    precision = tf.math.divide_no_nan(
      self.true_positives, self.true_positives + self.false_positives)
    recall = tf.math.divide_no_nan(
      self.true_positives, self.true_positives + self.false_negatives)

    mul_value = precision * recall
    add_value = (tf.math.square(self.beta) * precision) + recall
    mean = tf.math.divide_no_nan(mul_value, add_value)
    f1_score = mean * (1 + tf.math.square(self.beta))

    if self.average == "weighted":
      weights = tf.math.divide_no_nan(
        self.weights_intermediate, tf.reduce_sum(self.weights_intermediate))
      f1_score = tf.reduce_sum(f1_score * weights)

    elif self.average is not None:  # [micro, macro]
      f1_score = tf.reduce_mean(f1_score)

    return f1_score

  def reset_states(self):
    self.true_positives.assign(tf.zeros(self.init_shape, self.dtype))
    self.false_positives.assign(tf.zeros(self.init_shape, self.dtype))
    self.false_negatives.assign(tf.zeros(self.init_shape, self.dtype))
    self.weights_intermediate.assign(tf.zeros(self.init_shape, self.dtype))


class F1Score(FBetaScore):
  def __init__(self,
    num_classes: FloatTensorLike,
    average: str = None,
    threshold: Optional[FloatTensorLike] = None,
    name: str = "f1_score",
    dtype: AcceptableDTypes = None,
    **kwargs):
    super().__init__(num_classes, average, 1.0, threshold,
                     name=name, dtype=dtype)

from tensorflow.python.ops import math_ops

class SparseF1Score(F1Score):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.cast(math_ops.argmax(y_pred, axis=-1), self.dtype)
    return super().update_state(y_true, y_pred, sample_weight=sample_weight)

from stagedml.imports.tf import (INFINITE_CARDINALITY, UNKNOWN_CARDINALITY,
    cardinality)

def dataset_cardinality_size(d:Dataset)->Optional[int]:
  c=cardinality(d)
  if c==INFINITE_CARDINALITY:
    return None
  if c==UNKNOWN_CARDINALITY:
    return None
  return int(c)


def dataset_iter_size(d_fn:Callable[[],Dataset])->int:
  """ Returns Dataset cardinality if it is known, otherwize iterate and count.

  Notes:
  - Do not forget to take `batch_size` into account
  - The function will not finish if `Dataset.repeat()` was called
  """
  d=d_fn()
  c=dataset_cardinality_size(d)
  if c is not None:
    return c
  cnt=0
  for _ in d:
    cnt+=1
  return cnt

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE, DEFAULT_SIZE_GUIDANCE,
    ScalarEvent, TensorEvent )

def tensorboard_tags(rref:RRef,subfolder:str='train')->Dict[str,Union[list,bool]]:
  path=join(rref2path(rref),subfolder)
  event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
  event_acc.Reload()
  return event_acc.Tags()

def tensorboard_scalars(
    rref:RRef, subfolder:str, tag:str,
    scalar_guidance:int=10000)->List[ScalarEvent]:
  path=join(rref2path(rref),subfolder)
  event_acc=EventAccumulator(path, {'scalars': scalar_guidance})
  event_acc.Reload()
  return event_acc.Scalars(tag)

def tensorboard_tensors(rref:RRef, subfolder:str, tag:str,
                        tensor_guidance:int=10000)->List[TensorEvent]:
  path=join(rref2path(rref),subfolder)
  event_acc=EventAccumulator(path, {'tensors':tensor_guidance })
  event_acc.Reload()
  return event_acc.Tensors(tag)

def te2float(te:TensorEvent)->float:
  """ Should be a bug in TF """
  return float(MakeNdarray(te.tensor_proto).tolist()) # SIC!

