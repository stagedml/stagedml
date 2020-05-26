from pylightnix import ( Stage, Manager, Context, Hash, Path, DRef, RRef,
    Closure, Build, BuildArgs, Matcher, repl_realize, repl_continue, repl_build,
    build_outpath, realize, rref2path, store_config, config_name, mksymlink,
    isdir, dirhash, json_dump, json_load, assert_serializable,
    assert_valid_rref, build_wrapper_, readjson, store_rrefs, repl_rref,
    repl_cancel, rmref, store_gc, instantiate, tryreadjson, tryreadjson_def,
    mklens, Tag, RRefGroup, store_deps, store_initialize,
    assert_store_initialized )

from stagedml.imports import ( join, environ, remove, copytree, copy_tree,
    partial, AGraph, makedirs, History )
from stagedml.utils import ( runtensorboard, ndhashl )
from stagedml.types import ( Callable, List, Optional, Any, Tuple, Set,
    NamedTuple, Dict )

import tensorflow as tf

#: A default base for other global directories. Typically it is the root of the
#: local copy of the StagedML repository.
STAGEDML_ROOT=environ.get('STAGEDML_ROOT', environ.get('HOME','/tmp'))

#: Folder which has meaining for garbage collector. Symlinks to pylightnix
#: storage found here are preserved by the GC.
STAGEDML_EXPERIMENTS=environ.get('STAGEDML_EXPERIMENTS', STAGEDML_ROOT)

assert isdir(STAGEDML_ROOT), (
    f"StagedML root folder doesn't exist ('{STAGEDML_ROOT}'). Consider "
    f"assigning STAGEDML_ROOT environment variable to an existing direcory "
    f"path." )

#  _   _ _   _ _
# | | | | |_(_) |___
# | | | | __| | / __|
# | |_| | |_| | \__ \
#  \___/ \__|_|_|___/


def linkrrefs(rrefs:List[RRef], subdirs:List[str]=[], verbose:bool=False)->None:
  """ Create a 'result-...-N' symlink under the Pylightnix experiments folder """
  tgtpath_=Path(join(STAGEDML_EXPERIMENTS,*subdirs))
  for i,rref in enumerate(rrefs):
    suffix=f'-{i}' if len(rrefs)>1 else ''
    linkname=f'result-{config_name(store_config(rref))}{suffix}'
    if verbose:
      print(f"{tgtpath_}/{linkname} -> {rref}")
    if i==0:
      makedirs(tgtpath_, exist_ok=True)
    mksymlink(rref, tgtpath=tgtpath_, name=linkname, withtime=False)

def linkrref(rref:RRef, subdirs:List[str]=[], verbose:bool=False)->None:
  """ Create a 'result-' symlink under the Pylightnix experiments folder """
  linkrrefs([rref], subdirs, verbose)

def diskspace_h(sz:int)->str:
  return f"{sz//2**10} K" if sz<2**20 else \
         f"{sz//2**20} M" if sz<2**30 else \
         f"{sz//2**30} G"


#   ____
#  / ___|___  _ __ ___
# | |   / _ \| '__/ _ \
# | |__| (_) | | |  __/
#  \____\___/|_|  \___|


def assert_initialized()->None:
  assert isdir(STAGEDML_EXPERIMENTS), \
    (f"Looks like the StagedML experiments folder ('{STAGEDML_EXPERIMENTS}') "
     f"is not initialized. Did you call `initialize`?")
  assert_store_initialized()


def initialize(custom_pylightnix_store:Optional[str]=None,
               custom_pylightnix_tmp:Optional[str]=None,
               custom_stagedml_experiments:Optional[str]=None,
               check_not_exist:bool=False)->None:

  global STAGEDML_EXPERIMENTS
  store_initialize(custom_store=custom_pylightnix_store,
                   custom_tmp=custom_pylightnix_tmp,
                   check_not_exist=check_not_exist)
  if custom_stagedml_experiments is not None:
    STAGEDML_EXPERIMENTS=custom_stagedml_experiments
  makedirs(STAGEDML_EXPERIMENTS, exist_ok=True)
  assert_initialized()



def lrealize(clo:Closure, subdirs:List[str]=[], **kwargs)->RRef:
  """ Realize the model and Link it's realization to `STAGEDML_EXPERIMENTS` folder """
  assert_initialized()
  rref=realize(clo,**kwargs)
  linkrref(rref, subdirs)
  return rref


def borrow(rref:RRef, clo:Closure)->RRef:
  """ Borrows the contents of `rref` as-is to cook the realization of `clo`.
  Intended to hot-fix/monkey-patch realizations. Use with caution!

  - FIXME: Maybe move `borrow` to Pylightnix.
  - FIXME: Maybe re-define `borrow` via `pylightnix.redefine`+copying
    realizer.
  - FIXME: Maybe print config difference of source and target with
    `pylightnix.bashlike.diff`.
  """
  assert_initialized()
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
  orref=repl_continue(out_groups=[{Tag('out'):o}],rh=rh)
  assert orref is not None
  linkrref(orref)
  return orref


def tryrealize(clo:Closure, verbose:bool=False)->Optional[RRef]:
  """ Run the realization with all the realizers actually forbidden. Return
  non-empty value if local storage already contains all the realizations. """
  try:
    return realize(clo, assert_realized=[d.dref for d in clo.derivations])
  except Exception as e:
    if verbose:
      print(e)
    return None

def realize_recursive(step:Callable[[int,Optional[RRef]],Closure],
    epoches:int, epoches_step:int=1, force_rebuild:bool=False)->Dict[int,RRef]:
  rrefs:Dict[int,RRef]={}
  prev:Optional[RRef]=None
  for e in range(0,epoches,epoches_step):
    rrefs[e+epoches_step]=realize(step(e+epoches_step, prev), force_rebuild=force_rebuild)
    prev=rrefs[e+epoches_step]
  return rrefs

def depgraph(stages:List[Stage],
             filename:Optional[str]=None,
             layout:str='dot')->AGraph:
  """ Build a graph of dependencies for given stages. If `filename` is not
  None, save the graph into this file. """
  G=AGraph(strict=False,directed=True)
  touched:Set[DRef]=set()
  frontier=[instantiate(s).dref for s in stages]
  while len(frontier)>0:
    dref=frontier.pop()
    G.add_node(mklens(dref).name.val)
    for dep_dref in store_deps([dref]):
      G.add_node(mklens(dep_dref).name.val)
      G.add_edge(mklens(dref).name.val, mklens(dep_dref).name.val or dep_dref)
      if dep_dref not in touched:
        frontier.append(dep_dref)
      touched.add(dep_dref)

  if layout is not None:
    G.layout(prog=layout)
  if filename is not None:
    G.draw(filename)
  return G


#  ____            _                  _
# |  _ \ _ __ ___ | |_ ___   ___ ___ | |
# | |_) | '__/ _ \| __/ _ \ / __/ _ \| |
# |  __/| | | (_) | || (_) | (_| (_) | |
# |_|   |_|  \___/ \__\___/ \___\___/|_|


#: Protocol data, which is a list of protocol operations. An operation has a
#: name, a hash of whole model state and some payload data
ProtocolItem=NamedTuple('ProtocolItem',[('name',str),('hash',Optional[Hash]),('payload',Any)])
Protocol=List[ProtocolItem]

def protocol_dump(p:Protocol, fname:Path)->None:
  json=[{'name':i.name,'hash':i.hash,'payload':i.payload} for i in p]
  assert_serializable(json, argname='protocol')
  with open(fname,'w') as fp:
    json_dump(json, fp, indent=4)

def protocol_load(fname:Path)->Optional[Protocol]:
  json=tryreadjson(fname)
  return None if json is None else [ProtocolItem(p['name'],p['hash'],p['payload']) for p in json]

def protocol_load_def(fname:Path, default:Protocol)->Protocol:
  return protocol_load(fname) or default

def protocol_laststate(p:Protocol)->Optional[Hash]:
  return p[-1][1] if len(p)>0 else None

def protocol_add(fname:Path,
                 name:str,
                 whash:Optional[Hash]=None,
                 arg:Any=[],
                 result:Any=[],
                 expect_wchange:bool=True)->Protocol:
  """ Adds record to the protocol, saves protocol to disk """
  p=protocol_load_def(fname,[])
  old_whash=protocol_laststate(p)
  if whash is not None:
    if expect_wchange:
      assert whash != old_whash, \
          (f"Protocol sanity check: Operation was marked as parameter-changing,"
           f"but Model parameters didn't change their hashes as expected."
           f"Both hashes are {whash}.")
    else:
      assert whash == old_whash or (old_whash is None), \
          (f"Protocol sanity check: Operation was marked as"
           f"non-paramerer-changing, but Model parameters were in fact changed by"
           f"something. Expected {old_whash}, got {whash}.")
  p.append(ProtocolItem(name, whash, result))
  protocol_dump(p,fname)
  return p

def protocol_add_hist(fname:Path, name:str, whash:Hash, hist:History)->None:
  hd=hist.__dict__
  h2={'epoch':hd['epoch'],
      'history':{k:[float(f) for f in v] for k,v in hd['history'].items()}}
  protocol_add(fname, name, whash, result=h2)

def protocol_add_eval(fname:Path,
                      name:str,
                      whash:Hash,
                      metric_names:List[str],
                      result:List[float])->None:
  assert len(metric_names)==len(result), \
      f"{metric_names} doesn't match {result}"
  result=[float(x) for x in result]
  rec=[[a,b] for a,b in zip(metric_names,result)]
  protocol_add(fname, name, whash=whash, result=rec, expect_wchange=False)


# def store_protocol(rref:RRef)->Protocol:
#   assert_valid_rref(rref)
#   return list(readjson(join(rref2path(rref), 'protocol.json')))


#  __  __       _       _
# |  \/  | __ _| |_ ___| |__   ___ _ __ ___
# | |\/| |/ _` | __/ __| '_ \ / _ \ '__/ __|
# | |  | | (_| | || (__| | | |  __/ |  \__ \
# |_|  |_|\__,_|\__\___|_| |_|\___|_|  |___/


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


def best_(op_name:str, metric_name:str, rrefgs:List[RRefGroup])->RRefGroup:
  """ Return best model in terms of a metric, received by the given operation.
  Example: `best('evaluate','eval_accuracy', search(...)) ` """
  assert len(rrefgs)>0, "Empty input list of refs"
  metric_val=None
  best_rrefg=None
  for g in rrefgs:
    rref=g[Tag('out')]
    found_ops=0
    mv=None
    p=protocol_load(mklens(rref).protocol.syspath)
    if p is not None:
      mv=protocol_metric(p, op_name, metric_name)
    if mv is not None:
      if metric_val is None:
        metric_val=mv
        best_rrefg=g
      else:
        if mv>metric_val:
          metrci_val=mv
          best_rrefg=g
  assert best_rrefg is not None, \
    (f"`best()` was unable to find best match for '{metric_name}' "
     f"among '{op_name}' operations of {rrefgs}")
  return best_rrefg

def protocol_match(op_name:str, metric_name:str)->Matcher:
  def _matcher(dref:DRef, context:Context)->Optional[List[RRefGroup]]:
    rgs=list(store_rrefs(dref, context))
    if len(rgs)==0:
      return None
    return [best_(op_name, metric_name, rgs)]
  return _matcher

def protocol_rref_metric(rref:RRef,
                         op_name:str,
                         metric_name:str)->Optional[float]:
  p=protocol_load_def(mklens(rref).protocol.syspath, [])
  return protocol_metric(p, op_name, metric_name)

