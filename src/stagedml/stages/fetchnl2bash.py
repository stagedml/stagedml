from pylightnix import ( Hash, RefPath, Build, Path, Config, Manager, RRef,
    DRef, Context, build_wrapper, build_path, build_outpath, build_cattrs,
    mkdrv, rref2path, mkbuild, mkconfig, match_only, instantiate, realize,
    lsref, catref, store_cattrs, get_executable, dirhash, fetchlocal, mknode,
    mklens, instantiate, realize, repl_realize, repl_build, promise )

from stagedml.utils.files import ( system, flines )

from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile )

from stagedml.types import ( Optional,Any,List,Tuple,Union, WmtSubtok )
from stagedml.stages.files import ( splitfile )
from stagedml.stages.fetchwmt import ( wmtsubtok_ )


NL2BASH_ROOT=environ.get('NL2BASH_ROOT', join('/','workspace','3rdparty','nl2bash_essence', 'src'))

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


def nl2bashSubtok(m:Manager)->WmtSubtok:
  nl2b=fetchnl2bash(m)
  return wmtsubtok_(m,nl2b,nl2b,target_vocab_size=8192)
