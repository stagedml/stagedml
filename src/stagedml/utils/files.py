from os.path import islink
from json import load as json_load
from distutils.spawn import find_executable

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

