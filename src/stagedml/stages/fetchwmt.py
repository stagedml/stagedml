from pylightnix import ( RefPath, Build, Path, Config, Manager, RRef, DRef,
    Context, build_wrapper, build_path, build_outpath, build_cattrs, mkdrv,
    rref2path, mkbuild, mkconfig, match_only, instantiate, realize, lsref,
    catref, store_cattrs, get_executable, fetchurl, mknode, checkpaths, mklens,
    promise )

from stagedml.imports import ( join )
from stagedml.models.transformer.imports import ( Subtokenizer,
    encode_and_save_files )
from stagedml.utils.files import ( flines )
from stagedml.stages.files import catfiles
from stagedml.types import ( WmtSubtok, Optional, Any, List, Tuple, Union )

def fetchwmt17parallel(m:Manager)->DRef:
  langpairs=[['de','en'],['ru','en']]
  f=fetchurl(m,
    name='training-parallel-nc-v12',
    url='http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz',
    sha256='2b45f30ef1d550d302fd17dd3a5cbe19134ccc4c2cf50c2dae534aee600101a2')
  return checkpaths(m,
      {('_'.join(lp)):{l:[f,"training",f"news-commentary-v12.{'-'.join(lp)}.{l}"] \
      for l in lp} for lp in langpairs})

def fetchwmt13commoncrawl(m:Manager)->DRef:
  langpairs=[['de','en'],['ru','en']]
  f=fetchurl(m,
    name='training-parallel-commoncrawl',
    url='http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
    sha256='c7a74e2ea01ac6c920123108627e35278d4ccb5701e15428ffa34de86fa3a9e5')
  return checkpaths(m,
      {('_'.join(lp)):{l:[f,'training-parallel-commoncrawl',f"commoncrawl.{'-'.join(lp)}.{l}"] \
      for l in lp} for lp in langpairs})

def fetchwmt13europarl(m:Manager)->DRef:
  langpairs=[['de','en']]
  f=fetchurl(m,
    name='training-parallel-europarl-v7',
    url='http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
    sha256='0224c7c710c8a063dfd893b0cc0830202d61f4c75c17eb8e31836103d27d96e7')
  return checkpaths(m,
      {('_'.join(lp)):{l:[f,'training',f"europarl-v7.{'-'.join(lp)}.{l}"] \
      for l in lp} for lp in langpairs})

def fetchwmt17dev(m:Manager)->DRef:
  langs=['ru','en','de']
  f=fetchurl(m,
    name='newstest2013',
    url=f'http://data.statmt.org/wmt17/translation-task/dev.tgz',
    sha256='9d5ff04a28496b7796904ea65e50e3837bab14dbdca88b1e063105f17513dca9')
  return checkpaths(m,
    {l:[f,'dev',f'newstest2013.{l}'] for l in langs})

def fetchnewstest2014(m:Manager)->DRef:
  langs=['en','de']
  name='newstest2014'
  f=fetchurl(m,
    name=name,
    url=f'https://storage.googleapis.com/tf-perf-public/official_transformer/test_data/{name}.tgz',
    sha256='1d07bf20db2f4607bb5f3c228e24c1fa17bdfb66d650cc5e713353606fde0800')
  return checkpaths(m, {l:[f,'newstest2014',f'newstest2014.{l}'] for l in langs})


def fetchwmtpack(m:Manager)->DRef:
  return mknode(m,{
    'wmt17parallel':fetchwmt17parallel(m),
    'wmt13commoncrawl':fetchwmt13commoncrawl(m),
    'wmt13europarl':fetchwmt13europarl(m),
    'wmt17dev':fetchwmt17dev(m),
    'newstest2014':fetchnewstest2014(m),
    })


def trainfiles(m:Manager, lang1:str, lang2:str, europarl:Optional[bool]=None)->DRef:
  suffix=f"{lang2}_{lang1}"
  europarl_= europarl if europarl is not None else ('ru' not in [lang1,lang2])

  inputs=catfiles(m, outname='inputs', files=\
      [mklens(fetchwmt17parallel(m)).get(suffix).get(lang1).refpath,
       mklens(fetchwmt13commoncrawl(m)).get(suffix).get(lang1).refpath] + \
      ([mklens(fetchwmt13europarl(m)).get(suffix).get(lang1).refpath] if europarl_ else []))

  targets=catfiles(m, outname='targets', files=\
      [mklens(fetchwmt17parallel(m)).get(suffix).get(lang2).refpath,
       mklens(fetchwmt13commoncrawl(m)).get(suffix).get(lang2).refpath] + \
      ([mklens(fetchwmt13europarl(m)).get(suffix).get(lang2).refpath] if europarl_ else []))

  return checkpaths(m,{'name':f'wmt-{suffix}',
                       'train_input_combined':mklens(inputs).output.refpath,
                       'train_target_combined':mklens(targets).output.refpath})

def evalfiles(m:Manager, lang1:str, lang2:str)->DRef:
  suffix=f"{lang2}_{lang1}"
  inputs=catfiles(m, files=[mklens(fetchwmt17dev(m)).get(lang1).val], outname='inputs')
  targets=catfiles(m, files=[mklens(fetchwmt17dev(m)).get(lang2).val], outname='outputs')
  return mknode(m,{'name':f'wmt-{suffix}-eval',
                   'eval_input_combined':mklens(inputs).output.refpath,
                   'eval_target_combined':mklens(targets).output.refpath})


def wmtsubtok_(m:Manager, trainfiles:DRef, evalfiles:DRef, target_vocab_size:int=32768)->WmtSubtok:

  def _config():
    name = "subtok-"+mklens(trainfiles).name.val
    train_tag = "train"
    train_shards = 100
    eval_tag = "eval"

    # Link to the raw train data collection
    train_input_combined = mklens(trainfiles).train_input_combined.refpath
    train_target_combined = mklens(trainfiles).train_target_combined.refpath

    # Link to the raw eval data collection
    eval_input_combined = mklens(evalfiles).eval_input_combined.refpath
    eval_target_combined = mklens(evalfiles).eval_target_combined.refpath

    # Desired number of subtokens in the vocabulary list.
    nonlocal target_vocab_size
    # Accept vocabulary if size is within this threshold.
    target_threshold = 327
    # Minimum length of subtoken to pass the subtoken filter
    train_data_min_count:Optional[int] = 6
    # Vocabulry filename
    vocab_file = [promise, "vocab.%d" % target_vocab_size]
    return mkconfig({k:v for k,v in locals().items() if k!='m'})

  def _realize(b:Build):
    c=build_cattrs(b)
    o=build_outpath(b)
    train_combined=[build_path(b,c.train_input_combined), build_path(b,c.train_target_combined)]
    assert flines(train_combined[0])==flines(train_combined[1]), \
        "Numbers of lines in train files don't match. Consider checking line endings."
    eval_combined=[build_path(b,c.eval_input_combined), build_path(b,c.eval_target_combined)]
    assert flines(eval_combined[0])==flines(eval_combined[1]), \
        "Numbers of lines in eval files don't match. Consider checking line endings."
    subtokenizer = Subtokenizer.init_from_files(
        vocab_file=mklens(b).vocab_file.syspath,
        files=train_combined,
        target_vocab_size=c.target_vocab_size,
        threshold=c.target_threshold,
        min_count=c.train_data_min_count)
    print('Encoding train files')
    encode_and_save_files(subtokenizer, o, train_combined, c.train_tag, c.train_shards)
    print('Encoding eval files')
    encode_and_save_files(subtokenizer, o, eval_combined, c.eval_tag, 1)

  return WmtSubtok(mkdrv(m, _config(), match_only(), build_wrapper(_realize)))


def wmtsubtok(m:Manager, lang1:str, lang2:str)->WmtSubtok:
  return wmtsubtok_(m,trainfiles(m,lang1,lang2), evalfiles(m,lang1,lang2))

