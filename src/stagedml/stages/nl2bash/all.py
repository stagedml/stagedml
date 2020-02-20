"""
This module describes set of stages required to pre-process NL2BASH dataset

ref.  https://github.com/TellinaTool/nl2bash

Currently, it's HEAD commit `6663ca1` appears to be broken, so we extracted
some code from the earlier versions and tried to apply some fixes.

FIXME: Char-encoded data appears to be broker (every ID is '2')
"""

from typing import ( Optional,Any,List,Tuple,Union )

from pylightnix import ( RefPath, Build, Path, Config, Manager, RRef, DRef,
    Context, build_wrapper, build_path, build_outpath, build_cattrs, mkdrv,
    rref2path, mkbuild, mkconfig, match_only, instantiate, realize, lsref,
    catref, store_cattrs, get_executable )

from stagedml.utils.files import system

from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile )

from stagedml.types import NL2Bash

PYTHON=get_executable('python3', 'Python3 interpreter is required')
NL2BASH_ROOT=environ.get('NL2BASH_ROOT', join('/','workspace','3rdparty','nl2bash_essence', 'src'))

def run_nl2bash_env(nl2bash_root, cmd)->None:
  system(cmd,
    cwd=nl2bash_root,
    env={'PYTHONPATH':f'{nl2bash_root}:{environ["PYTHONPATH"]}'})

@contextmanager
def work_files(o:Path, files:List[str], remove:bool=True):
  for f in files:
    system(['cp', f, o])
    assert isfile(join(o,basename(f)))
  try:
    yield
  finally:
    if remove:
      for f in files:
        system(['rm', '-f', join(o,basename(f))])

@contextmanager
def work_refpaths(b:Build, refpaths:List[RefPath], remove:bool=True):
  with work_files(build_outpath(b), [build_path(b,p) for p in refpaths], remove):
    yield

def filter_data_config(nl2bash_root:str=NL2BASH_ROOT)->Config:
  name = 'nl2bash-filter-data'
  dataset = 'bash'
  num_utilities = 100
  use_nl2bash_essence = True
  return mkconfig(locals())

def filter_data_realize(b:Build)->None:
  o=build_outpath(b)
  c=build_cattrs(b)
  with work_files(o, [
    f'{c.nl2bash_root}/data/{c.dataset}/all.cm',
    f'{c.nl2bash_root}/data/{c.dataset}/all.nl',
    f'{c.nl2bash_root}/data/{c.dataset}/vocab.tar.xz']):

    run_nl2bash_env(c.nl2bash_root, [PYTHON, '-c', dedent(f'''
      from filter_data import *;
      filter_by_most_frequent_utilities("{o}", {c.num_utilities});
      ''')])

def filter_data(m:Manager)->DRef:
  return mkdrv(m, filter_data_config(), match_only(), build_wrapper(filter_data_realize))


def split_data_config(filtered_data:DRef):
  name = 'nl2bash-filtered-data'
  nl = [filtered_data,'all.nl.filtered']
  cm = [filtered_data,'all.cm.filtered']
  nl2bash_root = store_cattrs(filtered_data).nl2bash_root
  return mkconfig(locals())

def split_data_realize(b:Build)->None:
  o=build_outpath(b)
  c=build_cattrs(b)
  with work_refpaths(b, [c.nl, c.cm]):
    run_nl2bash_env(c.nl2bash_root, [PYTHON, '-c', dedent(f'''
      from split_data import *;
      split_data("{o}");
      ''')])


def split_data(m:Manager)->DRef:
  return mkdrv(m, split_data_config(filter_data(m)), match_only(), build_wrapper(split_data_realize))


def process_data_config(splitted_data:DRef)->Config:
  name = 'nl2bash-process-data'
  nl2bash_root = store_cattrs(splitted_data).nl2bash_root
  train_nl = [splitted_data, 'train.nl.filtered']
  dev_nl = [splitted_data, 'dev.nl.filtered']
  test_nl = [splitted_data, 'test.nl.filtered']
  train_cm = [splitted_data, 'train.cm.filtered']
  dev_cm = [splitted_data, 'dev.cm.filtered']
  test_cm = [splitted_data, 'test.cm.filtered']
  version = 2
  return mkconfig(locals())

def process_data_realize(b:Build)->None:
  o=build_outpath(b)
  c=build_cattrs(b)
  with work_refpaths(b, [c.train_nl, c.dev_nl, c.test_nl, c.train_cm, c.dev_cm, c.test_cm],
      remove=False):
    run_nl2bash_env(c.nl2bash_root, [PYTHON,
      '-m', 'process_data',
      '--rnn_cell', 'gru',
      '--encoder_topology', 'birnn',
      '--num_epochs', '100',
      '--num_samples', '256',
      '--variational_recurrent_dropout',
      '--token_decoding_algorithm', 'beam_search',
      '--beam_size', '100',
      '--alpha', '1.0',
      '--num_nn_slot_filling', '10',
      '--process_data',
      '--dataset', 'bash',
      '--data_dir', o])


def nl2bash(m:Manager)->NL2Bash:
  return NL2Bash(mkdrv(m,
    config=process_data_config(split_data(m)),
    matcher=match_only(),
    realizer=build_wrapper(process_data_realize)))

