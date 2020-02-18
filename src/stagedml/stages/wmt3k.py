
from typing import ( Optional,Any,List,Tuple,Union )

from pylightnix import ( RefPath, Build, Path, Config, Manager, RRef, DRef,
    Context, build_wrapper, build_path, build_outpath, build_cattrs, mkdrv,
    rref2path, mkbuild, mkconfig, match_only, instantiate, realize, lsref,
    catref, store_cattrs, get_executable )

from stagedml.utils.files import system

from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile )

PYTHON=get_executable('python3', 'Python3 interpreter is required')
TFM_ROOT=environ.get('TFM_ROOT', join('/','workspace','3rdparty','tensorflow_models'))

def run_tfm_env(tfm_root, cmd)->None:
  system(cmd,
    cwd=tfm_root,
    env={'PYTHONPATH':f'{tfm_root}:{environ["PYTHONPATH"]}'})

def wmt3kende_config()->Config:
  name = 'wmt3k_ende'
  return mkconfig(locals())

def wmt3kende_realize(b:Build)->None:
  o=build_outpath(b)
  c=build_cattrs(b)
  run_tfm_env(TFM_ROOT, [PYTHON,
    join(TFM_ROOT,'official','transformer', 'data_download.py'),
    '--data_dir', o])

def wmt3kende(m:Manager)->DRef:
  return mkdrv(m, wmt3kende_config(), match_only(), build_wrapper(wmt3kende_realize))
