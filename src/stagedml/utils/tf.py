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
from typing import ( Union, List, Any, Optional, Tuple, Callable, TypeVar )

from pylightnix import ( Closure, Path, Build, Hash, DRef, assert_valid_rref,
    assert_serializable, PYLIGHTNIX_TMP, Realizer, build_outpath, mkbuild, RRef,
    rref2path, readjson, json_dumps, store_rrefs, dirhash, Context,
    build_wrapper_, BuildArgs, repl_realize, repl_continue, repl_build, isrref )


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
  else:
    path=build_outpath(arg)
  pid=runtensorboard(path)
  print('Tensorboard is running at', path, 'pid', pid)
