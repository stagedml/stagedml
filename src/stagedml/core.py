
from pylightnix import ( Context, Hash, Path, DRef, RRef, Closure, Build,
    BuildArgs, repl_realize, repl_continue, repl_build, build_outpath, realize,
    rref2path, store_config, config_name, mksymlink, isdir, dirhash, json_dumps,
    assert_serializable, assert_valid_rref, build_wrapper_, readjson,
    store_rrefs, repl_rref, repl_cancel )

from stagedml.imports import ( join, environ, remove, copytree, copy_tree )
from stagedml.utils import ( runtensorboard, ndhashl )
from stagedml.types import ( Callable, List, Optional, Any, Tuple )

from stagedml.imports.tf import ( History )
import tensorflow as tf

STAGEDML_ROOT=environ.get('STAGEDML_ROOT', environ.get('HOME','/tmp'))
assert isdir(STAGEDML_ROOT), (
    f"StagedML root folder doesn't exist ('{STAGEDML_ROOT}'). Consider assigning "
    f"STAGEDML_ROOT environment variable to an existing direcory path." )


def linkrref(rref:RRef)->None:
  """ Create a 'result-' symlink under the Pylightnix root folder """
  mksymlink(rref, Path(STAGEDML_ROOT),
            name='result-'+config_name(store_config(rref)), withtime=False)


def lrealize(clo:Closure)->RRef:
  """ Realize the model and Link it's realization to STAGEDML_ROOT folder """
  rref=realize(clo)
  linkrref(rref)
  return rref


def borrow(rref:RRef, clo:Closure)->RRef:
  """ Borrows the contents of `rref` to cook the realization of `clo`.

  - FIXME: Maybe move `borrow` to Pylightnix.
  - FIXME: Maybe re-define `borrow` via `pylightnix.redefine`+copying
    realizer.
  - FIXME: Maybe print config difference of source and target with
    `pylightnix.bashlike.diff`.
  """
  rh=repl_realize(clo, force_interrupt=True)
  try:
    assert rh.drv is not None
    assert rh.dref is not None
    assert rh.context is not None
    b=repl_build(rh)
    o=build_outpath(b)
    copy_tree(rref2path(rref),o)
    remove(join(o,'context.json'))
  except Exception:
    repl_cancel(rh)
    raise
  orref=repl_continue(out_paths=[o],rh=rh)
  assert orref is not None
  linkrref(orref)
  return orref


def tryrealize(clo:Closure, verbose:bool=False)->Optional[RRef]:
  """ Run the realization, but deosn't allow it to run any realizers. """
  try:
    return realize(clo, assert_realized=[d.dref for d in clo.derivations])
  except Exception as e:
    if verbose:
      print(e)
    return None


# from pylightnix import datahash, encode, makedirs
# def linkrefs(rrefs:List[RRef], tgtdir:Optional[Path]=None)->Path:
#   if tgtdir is None:
#     import pylightnix.core
#     tgtdir=join(pylightnix.core.PYLIGHTNIX_TMP,
#                 datahash([encode(rref) for rref in rrefs])[:7])
#   makedirs(tgtdir,exist_ok=True)
#   for rref in rrefs:
#     mksymlink(rref,tgtdir,name=str(rref),withtime=False)
#   return tgtdir


#  ____        _ _     _
# | __ ) _   _(_) | __| | ___ _ __ ___
# |  _ \| | | | | |/ _` |/ _ \ '__/ __|
# | |_) | |_| | | | (_| |  __/ |  \__ \
# |____/ \__,_|_|_|\__,_|\___|_|  |___/


Protocol=List[Tuple[str,Hash,Any]]

class ProtocolBuild(Build):
  protocol:Protocol
  def __init__(self, ba:BuildArgs)->None:
    super().__init__(ba)
    self.protocol=[]
  def get_data_hash(self)->Hash:
    return dirhash(build_outpath(self))

def protocol_save(b:ProtocolBuild)->None:
  o=build_outpath(b)
  with open(join(o,'protocol.json'),'w') as f:
    f.write(json_dumps(b.protocol))

class KerasBuild(ProtocolBuild):
  model:tf.keras.Model
  def __init__(self, ba:BuildArgs)->None:
    super().__init__(ba)
  def get_data_hash(self)->Hash:
    assert self.model is not None, "Keras model should be initialized by the user"
    return Hash(ndhashl(self.model.get_weights()))

def keras_save(b:KerasBuild)->None:
  assert b.model is not None
  assert all(b.model._get_trainable_state().values())
  o = build_outpath(b)
  b.model.save_weights(join(o, 'weights.h5'), save_format='h5')
  protocol_save(b)

def protocolled(f:Callable[[ProtocolBuild],None], buildtime:bool=True):
  return build_wrapper_(f,ProtocolBuild,buildtime)

def keras_wrapper(f:Callable[[KerasBuild],None], buildtime:bool=True):
  return build_wrapper_(f,KerasBuild,buildtime)

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

def match_metric(op_name:str, metric_name:str):
  def _matcher(dref:DRef, context:Context)->Optional[List[RRef]]:
    rrefs=list(store_rrefs(dref, context))
    if len(rrefs)==0:
      return None
    return [best_(op_name, metric_name, rrefs)]
  return _matcher


