from stagedml.imports.sys import ( dedent )

from pylightnix import ( instantiate, realize, rref2path )

from .all import ( nl2bash, run_nl2bash_env, NL2BASH_ROOT, PYTHON )

def do():
  rref=realize(instantiate(nl2bash))
  print(rref)

  run_nl2bash_env(NL2BASH_ROOT, [PYTHON,
    "-c", dedent(f'''
      from process_data import *
      load_data(FLAGS)
      '''),
      '--dataset', 'bash',
      '--channel', 'token',
      '--data_dir', rref2path(rref)],
      )
