from os import environ,rename,remove
from os.path import join,isfile,basename
from typing import Optional,Any,List,Tuple,Union
from subprocess import Popen
from urllib.parse import urlparse
from hashlib import sha256

from pylightnix import ( Model, Config, State, Hash, Ref, model_save,
                         protocol_add, model_config_ro, model_outpath,
                         state_add, search, state )

from stagedml.utils.instantiate import Options, instantiate
from stagedml.utils.refs import Glue
from stagedml.datasets.glue.download_glue_data import main as glue_main



def config()->Config:
  name='glue-data'
  version='2'
  return Config({k:v for k,v in locals().items() if k[0]!='_'})


def downloaded(s:State)->State:
  return state_add(s, 'download')
def download(m:Model)->Model:
  """ Diagnostics is excluded here because it uses Google-drive, which is not
  available in places such as China. """
  c=model_config_ro(m)
  o=model_outpath(m)
  glue_main(['--data_dir', o, '--tasks','all'])
  protocol_add(m, 'download')
  return m

def fetchglue(o:Options)->Glue:
  c=config()
  def _search():
    return search(downloaded(state(c)))
  def _build():
    return model_save(download(Model(c)))
  return Glue(instantiate(o, _search, _build))

