from pylightnix import ( Manager, mknode, fetchurl, promise, mklens, get_executable )
from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile )
from stagedml.types import Enwiki

def fetchenwiki(m:Manager)->Enwiki:
  name='enwiki-20200301-pages-articles.xml.bz2'
  return Enwiki(fetchurl(m,
      name='fetchenwiki',
      url=f'https://dumps.wikimedia.org/enwiki/20200301/{name}',
      sha1='852dfec9eba3c4d5ec259e60dca233b6a777a05e',
      mode='asis',
      output=[promise,name]))
