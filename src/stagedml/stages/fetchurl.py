from os import environ,rename,remove
from os.path import join,isfile,basename
from typing import Optional,Any,List,Tuple,Union
from subprocess import Popen
from urllib.parse import urlparse
from hashlib import sha256

from pylightnix import ( Model, Config, State, Hash, Ref, model_save,
                         protocol_add, model_config_ro, model_outpath,
                         state_add, search, state )
from stagedml.utils.files import get_executable
from stagedml.utils.instantiate import Options, instantiate


WGET=get_executable('wget', 'Please install `wget` command')
AUNPACK=get_executable('aunpack', 'Please install `apack` tool from `atool` package')


def config(url:str, sha256:str, mode='unpack,remove', name=None)->Config:
  return Config({k:v for k,v in locals().items() if k[0]!='_'})


def downloaded(s:State)->State:
  return state_add(s, 'download')
def download(m:Model)->Model:
  c=model_config_ro(m)
  o=model_outpath(m)

  try:
    fname=basename(urlparse(c.url).path)
    partpath=join(o,fname+'.tmp')
    p=Popen([WGET, "--continue", '--output-document', partpath, c.url], cwd=o)
    p.wait()

    assert p.returncode == 0, f"Download failed, errcode '{p.returncode}'"
    assert isfile(partpath), f"Can't find output file '{partpath}'"

    with open(partpath,"rb") as f:
      realhash=sha256(f.read()).hexdigest();
      assert realhash==c.sha256, f"Expected sha256 checksum '{c.sha256}', but got '{realhash}' instead"

    fullpath=join(o,fname)
    rename(partpath, fullpath)

    if 'unpack' in c.mode:
      print("Unpacking {fullpath}..")
      p=Popen([AUNPACK, fullpath], cwd=o)
      p.wait()
      assert p.returncode == 0, f"Unpack failed, errcode '{p.returncode}'"
      if 'remove' in c.mode:
        print("Removing {fullpath}..")
        remove(fullpath)

    protocol_add(m, 'download')
  except Exception as e:
    print(f"Download failed:",e)
    print(f"Temp folder {o}")
    raise
  return m


def fetchurl(o:Options, *args, **kwargs)->Ref:
  c=config(*args, **kwargs)
  def _search():
    return search(downloaded(state(c)))
  def _build():
    return model_save(download(Model(c)))
  return instantiate(o, _search, _build)
