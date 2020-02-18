from stagedml.imports import ( dedent )

from pylightnix import ( instantiate, realize, rref2path )

from .all import ( nl2bash, run_nl2bash_env, NL2BASH_ROOT, PYTHON )

from process_data import FLAGS, load_data

import sys

sys.argv=['ipython']
rref=realize(instantiate(nl2bash))

def data():
  print(rref)
  return rref

# def del_all_flags(FLAGS):
#   flags_dict = FLAGS._flags()
#     keys_list = [keys for keys in flags_dict]
#     for keys in keys_list:
#       FLAGS.__delattr__(keys)

def test():
  # FLAGS.remove_flag_values(['profile-dir'])
  FLAGS.dataset='bash'
  FLAGS.channel='token'
  FLAGS.data_dir=str(rref2path(data()))
  print(FLAGS.data_dir)
  return load_data(FLAGS, use_buckets=False)

