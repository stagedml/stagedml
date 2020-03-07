from pylightnix import ( Hash, RefPath, Build, Path, Config, Manager, RRef,
    DRef, Context, build_wrapper, build_path, build_outpath, build_cattrs,
    mkdrv, rref2path, mkbuild, mkconfig, match_only, instantiate, realize,
    lsref, catref, store_cattrs, get_executable, dirhash, fetchlocal, mknode,
    mklens, instantiate, realize, repl_realize, repl_build, promise )

from stagedml.utils.files import ( system, flines, writelines, readlines )

from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile )

from stagedml.types import ( Optional,Any,List,Tuple,Union, WmtSubtok, Set )
from stagedml.stages.files import ( splitfile )
from stagedml.stages.fetchwmt import ( wmtsubtok_ )

from official.nlp.transformer.utils.tokenizer import ( EOS_ID, Subtokenizer,
    alphanumeric_char_set, RESERVED_TOKENS )

BASH_CHAR_SET:Set[Any] = alphanumeric_char_set() | set(['-','+',',','.'])

def non_alphanumeric_chars(filepath:str)->Set[str]:
  ret=set()
  with open(filepath,'r') as f:
    for (i,line) in enumerate(f):
      for c in line.strip():
        if c not in BASH_CHAR_SET:
          ret.add(c)
  return ret

def cathegories(fp:str)->List[Set[Any]]:
  return [BASH_CHAR_SET] + [set([c]) for c in non_alphanumeric_chars(fp)]

def whichcat(c:str, cats:List[Set[Any]])->int:
  for i,cat in enumerate(cats):
    if c in cat:
      return i
  assert False, f"Unknown char found: {c}. This shouldn't be possible."


def simpleparse(fp:str)->List[Tuple[str,List[str]]]:
  ret=[]
  cats=cathegories(fp)
  with open(fp,'r') as f:
    for (_,line) in enumerate(f):
      line=line.strip()
      curcat=None
      start=0
      toks:list=[]
      for (i,c) in enumerate(line):
        newcat=whichcat(c,cats)
        if curcat is None:
          curcat=newcat
        else:
          if newcat==curcat and curcat==0:
            pass
          else:
            toks.append(line[start:i])
            start=i
            curcat=newcat
      toks.append(line[start:])
      ret.append((line,toks))
  return ret

def collect_bash_specific_tokens(fp:str)->Set[str]:
  ret=set()
  tlines=simpleparse(fp)
  for (_,tline) in tlines:
    for i,t in enumerate(tline):
      if i==0: # Command name
        ret.add(t)
      if t[0] not in BASH_CHAR_SET: # Special symbol
        ret.add(t)
      if t[0]=='-': # A flag
        ret.add(t)
      pass
  return ret


def fetchnl2bash(m:Manager)->DRef:
  def _split(m, fn_suffix:str, sha256:str ):
    raw=fetchlocal(m, path=join('3rdparty','nl2bash_essence','src','data','bash',f'all.{fn_suffix}'),
                      sha256=sha256, mode='asis',
                      output=[promise, f'all.{fn_suffix}'] )
    split=splitfile(m, src=mklens(raw).output.refpath,
                       fractions=[('train',f'train_{fn_suffix}.txt', 0.9),
                                  ('eval', f'eval_{fn_suffix}.txt',0.1)])
    return split

  nlfiles=_split(m, 'nl', sha256='1db0c529c350b463919624550b8f5882a97c42ad5051c7d49fbc496bc4e8b770')
  cmfiles=_split(m, 'cm', sha256='3a72eaced7fa14a0938354cefc42b2dcafb2d47297102f1279086e18c3abe57e')

  return mknode(m, {
    'name':'fetchnl2bash',
    'train_input_combined':mklens(nlfiles).train.refpath,
    'train_target_combined':mklens(cmfiles).train.refpath,
    'eval_input_combined':mklens(nlfiles).eval.refpath,
    'eval_target_combined':mklens(cmfiles).eval.refpath
    })


def bash_reserved_tokens(m:Manager, inputfile:RefPath)->DRef:
  config=mkconfig({'inp':inputfile, 'output':[promise,'reserved_tokens.txt']})
  def _realize(b:Build)->None:
    o=build_outpath(b)
    reserved_tokens=list(collect_bash_specific_tokens(mklens(b).inp.syspath))
    writelines(mklens(b).output.syspath, reserved_tokens)
    test=list(readlines(mklens(b).output.syspath))
    assert reserved_tokens==test, f"{reserved_tokens[0]},.. != {test[0]},.."
  return mkdrv(m, config, match_only(), realizer=build_wrapper(_realize))


def bash_charset(m:Manager)->DRef:
  config=mkconfig({'name':'bash_charset','output':[promise,'charset.txt']})
  def _realize(b:Build)->None:
    o=build_outpath(b)
    charset=list(BASH_CHAR_SET)
    writelines(mklens(b).output.syspath, charset)
    test=list(readlines(mklens(b).output.syspath))
    assert test==charset
  return mkdrv(m, config, match_only(), realizer=build_wrapper(_realize))

def nl2bashSubtok(m:Manager)->WmtSubtok:
  nl2b=fetchnl2bash(m)
  restok=mklens(bash_reserved_tokens(m,mklens(nl2b).train_target_combined.refpath)).output.refpath
  charset=mklens(bash_charset(m)).output.refpath
  return wmtsubtok_(m,nl2b,nl2b,
                      target_vocab_size=8192,
                      train_shards=1,
                      reserved_tokens=restok,
                      master_char_set=charset)

