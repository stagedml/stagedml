from os import environ,rename,remove
from os.path import join,isfile,basename
from typing import Optional,Any,List,Tuple,Union
from subprocess import Popen
from urllib.parse import urlparse
from hashlib import sha256

from pylightnix import ( Config, Hash, DRef, build_cattrs, build_outpath,
    Manager, only, mkdrv, build_wrapper, Build )

from stagedml.utils.refs import Glue
from stagedml.datasets.glue.download_glue_data import main as glue_main


def config()->Config:
  name='glue-data'
  version='2'
  return Config(locals())


def download(b:Build)->None:
  glue_main(['--data_dir', build_outpath(b), '--tasks','all'])


def fetchglue(m:Manager)->Glue:
  def _instantiate():
    return config()
  def _match():
    return only()
  def _realize():
    return build_wrapper(download)
  return Glue(mkdrv(m, _instantiate, _match(), _realize()))

