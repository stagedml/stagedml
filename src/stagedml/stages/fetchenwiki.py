from pylightnix import ( Manager, mknode, fetchurl, promise, mklens,
    get_executable, DRef, Build, mkdrv, match_only, build_wrapper,
    build_setoutpaths, promise, build_outpath, mkconfig )
from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile, find_executable, cpu_count )
from stagedml.utils import system
from stagedml.types import Wikidump, Wikitext, Path, List, Optional

def fetchwiki(m:Manager, dumpname:str, dumpdate:str, sha1:str)->Wikidump:
  name=f'{dumpname}-{dumpdate}-pages-articles.xml.bz2'
  return Wikidump(fetchurl(m,
      name='fetchenwiki',
      url=f'https://dumps.wikimedia.org/{dumpname}/20200301/{name}',
      sha1=sha1,
      mode='asis',
      output=[promise,name]))

WIKIEXTRACTOR=find_executable('WikiExtractor.py')

WIKIEXTRACTOR_ARGS=[
  # '--links',
  # '--sections',
  # '--lists',
  # '--keep_tables',
  '--min_text_length', '0',
  '--filter_disambig_pages',
  ]

def extractwiki_realize(b:Build)->None:
  assert WIKIEXTRACTOR is not None, (
    "WikiExtractor.py is not found in PYTHONPATH, please install it" )
  build_setoutpaths(b,1)
  system(
    cmd=([WIKIEXTRACTOR,
        mklens(b).wikiref.syspath,
        '--json',
        '--processes', str(max(1,((cpu_count() or 1)*3)//4)),
        '--output', mklens(b).output.syspath,
        '--compress',
        '--bytes', f'{mklens(b).filesize_mb.val}M',
        # '--log_file', join(build_outpath(b),'wikiextractor.log'),
        ]) + \
        (['--templates', mklens(b).templates.syspath]
             if mklens(b).templates.optval else ['--no_templates']) + \
        (mklens(b).wikiextractor_args.val))

def extractwiki(m:Manager,
                wikiref:Wikidump,
                with_templates:bool=False,
                wikiextractor_args:Optional[List[str]]=None)->Wikitext:
  assert WIKIEXTRACTOR is not None, (
    "Can't find `WikiExtractor.py` executable! You can install it from "
    "https://github.com/stagedml/wikiextractor" )
  wa=wikiextractor_args if wikiextractor_args is not None \
                        else WIKIEXTRACTOR_ARGS

  config={
    'name':'extractwiki',
    'wikiref':mklens(wikiref).output.refpath,
    'templates':[promise,'templates.txt'] if with_templates else None,
    'output':[promise,'output'],
    'filesize_mb': 100,
    'wikiextractor_args':wa,
    }

  return Wikitext(mkdrv(m,mkconfig(config),
                          match_only(),
                          build_wrapper(extractwiki_realize)))

