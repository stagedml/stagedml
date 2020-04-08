from pylightnix import ( Manager, mknode, fetchurl, promise, mklens,
    get_executable, DRef, Build, mkdrv, match_only, build_wrapper,
    build_setoutpaths, promise, build_outpath, mkconfig )
from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile, find_executable )
from stagedml.utils import system
from stagedml.types import Wikidump, Wikitext

def fetchwiki(m:Manager, dumpname:str, dumpdate:str, sha1:str)->Wikidump:
  name=f'{dumpname}-{dumpdate}-pages-articles.xml.bz2'
  return Wikidump(fetchurl(m,
      name='fetchenwiki',
      url=f'https://dumps.wikimedia.org/{dumpname}/20200301/{name}',
      sha1=sha1,
      mode='asis',
      output=[promise,name]))

WIKIEXTRACTOR=find_executable('WikiExtractor.py')


def extractwiki_realize(b:Build)->None:
  assert WIKIEXTRACTOR is not None
  build_outpath(b)
  system([WIKIEXTRACTOR,
     mklens(b).wiki.syspath,
     '--json',
     '--processes', str(15),
     '--templates', mklens(b).templates.syspath,
     '--output', mklens(b).output.syspath,
     '--bytes', '1M',
     '--compress',
     # '--links',
     '--sections',
     '--lists',
     # '--keep_tables',
     '--min_text_length', '0',
     '--filter_disambig_pages'])

def extractwiki(m:Manager, wiki:Wikidump)->Wikitext:
  assert WIKIEXTRACTOR is not None, "Can't find `WikiExtractor.py` executable!"
  config={
    'name':'extractwiki',
    'wiki':mklens(wiki).output.refpath,
    'templates':[promise,'templates.txt'],
    'output':[promise,'output']
    }

  return Wikitext(mkdrv(m,mkconfig(config),match_only(),build_wrapper(extractwiki_realize)))
