from pylightnix import ( Manager, mknode, fetchurl, promise, mklens,
    get_executable, DRef, Build, mkdrv, match_only, build_wrapper,
    build_setoutpaths, promise, build_outpath, mkconfig )
from stagedml.imports.sys import ( environ, join, basename, dedent,
    contextmanager, isfile, find_executable, cpu_count, bz2_open, walk, Pool,
    abspath, json_loads )
from stagedml.utils.files import ( system, writestr )
from stagedml.types import ( Wikidump, Wikitext, Path, List, Optional, Tuple )

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
        (['--templates', mklens(b).templates.syspath] \
          if mklens(b).templates.optval else ['--no_templates']) + \
        (mklens(b).wikiextractor_args.val))

def extractwiki(m:Manager,
                wikiref:Wikidump,
                with_templates:bool=False,
                wikiextractor_args:Optional[List[str]]=None)->Wikitext:
  """ Extract Wikipedia dump into a set of compressed JSON files.

  FIXME: Add promise with final command line of WikiExtractor
  """
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

  return Wikitext(mkdrv(m, mkconfig(config), match_only(),
                           build_wrapper(extractwiki_realize)))

def wikistat_process_file(input_file:str)->Tuple[int,int]:
  nwords=0; ndocs=0
  with bz2_open(input_file, "rt", encoding='utf-8') as reader:
    for json_doc in reader:
      sentences=json_loads(json_doc)['text'].split('\n')
      nwords+=sum([len(s.split()) for s in sentences])
      ndocs+=1
  return ndocs,nwords

def wikistat_realize(b):
  build_setoutpaths(b,1)

  input_files = []
  for root, dirs, filenames in walk(mklens(b).input_folder.syspath, topdown=True):
    for filename in sorted(filenames):
      if filename.endswith('bz2'):
        input_files.append(abspath(join(root, filename)))

  with Pool() as p:
    ndocs,nwords=zip(*list(p.map(wikistat_process_file, input_files)))

  writestr(mklens(b).output.docnum.syspath, str(sum(ndocs)))
  writestr(mklens(b).output.wordnum.syspath, str(sum(nwords)))


def wikistat(m:Manager,
             wikiref:Wikitext)->DRef:
  def _config():
    name='wikistat'
    nonlocal wikiref
    input_folder=mklens(wikiref).output.refpath
    output={'docnum':[promise,'docnum.txt'],
            'wordnum':[promise,'wordnum.txt']}
    version=5
    return locals()

  return mkdrv(m, mkconfig(_config()), match_only(),
                  build_wrapper(wikistat_realize))

