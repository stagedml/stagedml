from stagedml.imports.sys import (mkconfig, Manager, match_only, mkdrv,
                                  build_wrapper, Build, build_setoutpaths,
                                  mklens, promise,

                                  makedirs, randint)

from stagedml.types import Glue


def gen_glue(m:Manager, task_name:str)->Glue:
  """ Generate dataset which looks just like task `TASK_NAME` of GLUE.
  Currenlty, only one task is supported """
  def _config():
    name = 'glue-generated'
    nonlocal task_name
    if task_name != 'MRPC':
      raise ValueError(f"Task name {task_name} is not supported")
    task_dir = [promise, task_name]
    task_train = task_dir + ['train.tsv']
    task_dev = task_dir + ['dev.tsv']
    task_test = task_dir + ['test.tsv']
    train_size = 100
    return locals()

  def _make(b:Build):
    build_setoutpaths(b, 1)
    makedirs(mklens(b).task_dir.syspath)
    with open(mklens(b).task_train.syspath, 'w') as f:
      for i in range(mklens(b).train_size.val):
        source = 'gj04'
        label = '1' if randint(0,1)>0 else '1'
        star = '*' if randint(0,1)>0 else ' '
        sentence = 'Mary had a little lamb.'
        f.write(f"{source} {label} {star} {sentence}.\n")

    with open(mklens(b).task_dev.syspath, 'w') as f:
      # FIXME
      pass
    with open(mklens(b).task_test.syspath, 'w') as f:
      # FIXME
      pass

  return Glue(mkdrv(m, mkconfig(_config()), match_only(), build_wrapper(_make)))

