import logging
from stagedml.imports.sys import ( find_executable, Popen, json_load, islink,
    chain )
from typing import ( List, Any, Optional, Dict, Iterable, Iterator )


def tryindex(l:list,item)->Optional[int]:
  try:
    return l.index(item)
  except ValueError:
    return None

def concat(l:List[List[Any]])->List[Any]:
  return list(chain.from_iterable(l))

def batch(ls:List[Any], n:int=1)->Iterator[List[Any]]:
  l=len(ls)
  for ndx in range(0, l, n):
    yield ls[ndx:min(ndx + n, l)]

