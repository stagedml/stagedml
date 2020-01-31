""" Utility functions which require TensorFlow as a dependency """
import json
import numpy
import tensorflow as tf
assert tf.version.VERSION.startswith('2.1'), \
       f"sbtagedml requires TensorFlow version '2.1.*', not '{tf.version.VERSION}'"

from re import search as re_search
from os import remove, listdir
from os.path import join
from tensorflow.keras.callbacks import History
from hashlib import md5
from subprocess import run as os_run, Popen
from pylightnix import ( Build, Hash, DRef, assert_valid_rref,
    assert_serializable, PYLIGHTNIX_TMP, Realizer, build_outpath, mkbuild,
    RRef, rref2path, readjson, store_rrefs, dirhash, Context )
from typing import ( Union, List, Any, Optional, Tuple, Callable )


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

def ndhashl(arrays:List[numpy.array])->str:
  e=md5()
  for l in arrays:
    e.update(l)
  return e.hexdigest()

def ndhash(a:numpy.array)->str:
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


Protocol=List[Tuple[str,Hash,Any]]


class ProtocolBuild(Build):
  protocol:Protocol
  def __init__(self, b:Build)->None:
    super().__init__(b.dref, b.context, b.timeprefix, b.outpath)
    self.protocol=[]
  def get_data_hash(self)->Hash:
    return dirhash(build_outpath(self))


def protocolled(f:Callable[[ProtocolBuild],None], buildtime:bool=True)->Realizer:
  def _wrapper(dref,context):
    pb=ProtocolBuild(mkbuild(dref,context,buildtime)); f(pb); return build_outpath(pb)
  return _wrapper


class KerasBuild(ProtocolBuild):
  model:tf.keras.Model
  def __init__(self, b:Build)->None:
    super().__init__(b)
  def get_data_hash(self)->Hash:
    assert self.model is not None, "Keras model should be initialized by the user"
    return Hash(ndhashl(self.model.get_weights()))

def keras_realizer(f:Callable[[KerasBuild],None], buildtime:bool=True)->Realizer:
  def _wrapper(dref,context):
    kb=KerasBuild(mkbuild(dref,context,buildtime)); f(kb); return build_outpath(kb)
  return _wrapper

#  ____            _                  _
# |  _ \ _ __ ___ | |_ ___   ___ ___ | |
# | |_) | '__/ _ \| __/ _ \ / __/ _ \| |
# |  __/| | | (_) | || (_) | (_| (_) | |
# |_|   |_|  \___/ \__\___/ \___\___/|_|


def protocol_laststate(b:ProtocolBuild)->Optional[Hash]:
  if len(b.protocol) == 0:
    return None
  else:
    return b.protocol[-1][1]

def protocol_add(build:ProtocolBuild, name:str, arg:Any=[], result:Any=[], expect_wchange:bool=True)->None:
  assert_serializable(name,'name')
  assert_serializable(arg,'arg')
  assert_serializable(result,'result')
  new_whash=build.get_data_hash()
  old_whash=protocol_laststate(build)
  if expect_wchange:
    assert new_whash != old_whash, \
        (f"Protocol sanity check: Operation was marked as parameter-changing,"
         f"but Model parameters didn't change their hashes as expected."
         f"Both hashes are {new_whash}.")
  else:
    assert new_whash == old_whash or (old_whash is None), \
        (f"Protocol sanity check: Operation was marked as"
         f"non-paramerer-changing, but Model parameters were in fact changed by"
         f"something. Expected {old_whash}, got {new_whash}.")
  build.protocol.append((name, new_whash, result))

def protocol_add_hist(build:ProtocolBuild, name:str, hist:History)->None:
  hd=hist.__dict__
  h2={'epoch':hd['epoch'],
      'history':{k:[float(f) for f in v] for k,v in hd['history'].items()}}
  protocol_add(build, name, result=h2)

def protocol_add_eval(build:ProtocolBuild, name:str, metric_names:List[str], result:List[float])->None:
  result=[float(x) for x in result]
  rec=[[a,b] for a,b in zip(metric_names,result)]
  protocol_add(build, name, result=rec, expect_wchange=False)












# def save(m:KerasBuild)->Ref:
#   assert all(m.model._get_trainable_state().values())
#   o = model_outpath(m)
#   m.model.save_weights(join(o, 'weights.h5'), save_format='h5')
#   r = model_save(m)
#   return r

def runtensorboard(path:str, kill_existing:bool=True)->int:
  if kill_existing:
    os_run('ps fax | grep -v grep | grep tensorboard | awk "{print \$1}" | xargs -r kill', shell=True)
  with open(join(PYLIGHTNIX_TMP,"tensorboard.log"),"w") as f:
    pid = Popen(["tensorboard", "--host", "0.0.0.0", "--logdir", path],
                stdout=f, stderr=f).pid
  return pid

def runtb(arg:Union[Build,str])->None:
  if isinstance(arg,str):
    path=arg
    pid=runtensorboard(path)
    print('Tensorboard is running at', path, 'pid', pid)
  else:
    path=build_outpath(arg)
    pid=runtensorboard(path)
    print('Tensorboard is running at', path, 'pid', pid)

def store_protocol(rref:RRef)->Protocol:
  assert_valid_rref(rref)
  return list(readjson(join(rref2path(rref), 'protocol.json')))

def protocol_metric(p:Protocol, op_name:str, metric_name:str)->Optional[float]:
  found_ops=0
  metric_val=None
  for (n,h,metrics) in reversed(p):
    if n==op_name:
      found_ops+=1
      found_metrics=0
      for (mname,mval) in metrics:
        if mname==metric_name:
          found_metrics+=1
          if metric_val is None:
            metric_val=mval
          else:
            if mval>metric_val:
              metric_val=mval
      if found_metrics==0:
        print(f"Warning: '{metric_name}' metric was not found for op '{n}'")
      break
  if found_ops==0:
    print(f"Warning: '{op_name}' operation was found in protocol")
  return metric_val

def best_(op_name:str, metric_name:str, refs:List[RRef])->RRef:
  """ Return best model in terms of a metric, received by the given operation.
  Example: `best('evaluate','eval_accuracy', search(...)) ` """
  assert len(refs)>0, "Empty input list of refs"
  metric_val=None
  best_ref=None
  for ref in refs:
    p=store_protocol(ref)
    found_ops=0
    mv=protocol_metric(p, op_name, metric_name)
    if mv is not None:
      if metric_val is None:
        metric_val=mv
        best_ref=ref
      else:
        if mv>metric_val:
          metrci_val=mv
          best_ref=ref
  assert best_ref is not None, \
    (f"`best()` was unable to find best match for '{metric_name}' "
     f"among '{op_name}' operations")
  return best_ref

def bestmatch(op_name:str, metric_name:str):
  def _matcher(dref:DRef, context:Context)->Optional[RRef]:
    return best_(op_name, metric_name, list(store_rrefs(dref, context)))
  return _matcher


