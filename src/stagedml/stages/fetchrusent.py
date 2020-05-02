from pylightnix import ( Hash, RefPath, Build, Path, Config, Manager, RRef,
    DRef, Context, build_wrapper, build_path, build_outpath, build_cattrs,
    mkdrv, rref2path, mkbuild, mkconfig, match_only, instantiate, realize,
    lsref, catref, store_cattrs, dirhash, fetchlocal, mknode,
    mklens, instantiate, realize, repl_realize, repl_build, promise )

from stagedml.core import STAGEDML_ROOT
from stagedml.utils.files import ( system, flines, writelines, readlines )

from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile )

from stagedml.types import ( Optional, Any, List, Tuple, Union, Rusent, Set )


def fetchrusent(m:Manager, shuffle:bool=True)->Rusent:
  return Rusent(fetchlocal(m,
    name='fetchrusent',
    envname='STAGEDML_RUSENTIMENT',
    sha256='cbc02dfbfaee81eda1f192b5280f05fbda41fb1ab9952cb4d8f7b0ff227c968d',
    output_preselected=[promise, 'rusentiment.tar', 'rusentiment_preselected_posts.csv'],
    output_random=[promise, 'rusentiment.tar', 'rusentiment_random_posts.csv'],
    output_tests=[promise, 'rusentiment.tar', 'rusentiment_test.csv']))


from pandas import DataFrame, read_csv

def created_tfrecords(filepath:str)->Any:
  pd1=read_csv(filepath)
  print(pd1)
  return pd1

def rusent_process(b:Build)->None:
  c=build_cattrs(b)
  o=build_outpath(b)

  created_tfrecords(mklens(b).refdataset.output_preselected.syspath)

  # makedirs(task_out_dir)
  # create_tfrecord_data(task_name=c.task_name,
  #                      data_dir=mklens(b).inputdir.syspath,
  #                      vocab_path=mklens(b).bert_vocab.syspath,
  #                      output_dir=task_out_dir,
  #                      max_seq_length=c.max_seq_length)

def rusent_tfrecords(m:Manager,
                     bert_vocab:RefPath,
                     refdataset:Rusent)->DRef:
  def _config():
    name='tfrecord-rusent'
    nonlocal bert_vocab
    nonlocal refdataset
    outputs={'train':[promise,'train.tfrecord'],
             'dev':[promise,'dev.tfrecord'],
             'meta':[promise,'meta.json']}
    max_seq_length = 128
    return locals()

  return Rusent(
    mkdrv(m,
      config=mkconfig(_config()),
      matcher=match_only(),
      realizer=build_wrapper(rusent_process)))


