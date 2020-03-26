import logging
from stagedml.imports import ( find_executable, Popen, json_load, islink )
from typing import List, Any, Optional, Dict, Iterable

def listloggers()->List[Any]:
  l=logging.root.manager.loggerDict # type:ignore
  return [logging.getLogger(name) for name in l]

def json_read(filename:str)->dict:
  with open(filename,"r") as f:
    return json_load(f)

def get_executable(name:str, not_found_message:str)->str:
  e=find_executable(name)
  assert e is not None, not_found_message
  return e

def assert_link(name:str, not_found_message:str)->None:
  if not islink(name):
    assert False, not_found_message

def system(cmd:List[str], cwd:Optional[str]=None, env:Optional[dict]=None,
           check_retcode:bool=True)->None:
  args:Dict[str,Any]={}
  if cwd is not None:
    args.update({'cwd':cwd})
  if env is not None:
    args.update({'env':env})
  p=Popen(cmd, **args)
  retcode=p.wait()
  assert not (check_retcode and retcode!=0), f"Retcode is not zero, but {retcode}"
  return

def readlines(filepath:str, tostrip:str='\n')->Iterable[str]:
  with open(filepath,'r') as f:
    for line in f:
      line_stripped=line.rstrip(tostrip)
      assert tostrip not in line_stripped
      yield line_stripped

def writelines(filepath:str, lines:Iterable[str])->None:
  with open(filepath,'w') as f:
    for line in lines:
      assert '\n' not in line, f"Can't save line '{line}' because it contains EOL"
      f.write(line); f.write('\n')


def flines(p:str, newline:str='\n')->int:
  with open(p,'r',newline=newline) as f:
    for i, l in enumerate(f):
      pass
  return i+1


def tryindex(l:list,item)->Optional[int]:
  try:
    return l.index(item)
  except ValueError:
    return None

