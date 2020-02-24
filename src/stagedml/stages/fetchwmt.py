from pylightnix import ( RefPath, Build, Path, Config, Manager, RRef, DRef,
    Context, build_wrapper, build_path, build_outpath, build_cattrs, mkdrv,
    rref2path, mkbuild, mkconfig, match_only, instantiate, realize, lsref,
    catref, store_cattrs, get_executable, fetchurl, mknode )

from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile )

from stagedml.models.transformer.imports import ( Subtokenizer,
    encode_and_save_files )

from stagedml.utils.files import ( system, flines )

from stagedml.types import ( WmtTfrecs, Optional, Any, List, Tuple, Union )


# PYTHON=get_executable('python3', 'Python3 interpreter is required')
# TFM_ROOT=environ.get('TFM_ROOT', join('/','workspace','3rdparty','tensorflow_models'))

# def run_tfm_env(tfm_root, cmd)->None:
#   system(cmd,
#     cwd=tfm_root,
#     env={'PYTHONPATH':f'{tfm_root}:{environ["PYTHONPATH"]}'})

# def wmt32ende_config()->Config:
#   name = 'wmt3k_ende'
#   return mkconfig(locals())

# def wmt32ende_realize(b:Build)->None:
#   o=build_outpath(b)
#   c=build_cattrs(b)
#   run_tfm_env(TFM_ROOT, [PYTHON,
#     join(TFM_ROOT,'official','transformer', 'data_download.py'),
#     '--data_dir', o])

# def wmt32ende(m:Manager)->Wmt:
#   return Wmt(mkdrv(m, wmt32ende_config(),
#                       match_only(),
#                       build_wrapper(wmt32ende_realize)))



def fetchwmt17parallel(m:Manager)->DRef:
  return fetchurl(m,
    name='training-parallel-nc-v12',
    url='http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz',
    sha256='2b45f30ef1d550d302fd17dd3a5cbe19134ccc4c2cf50c2dae534aee600101a2')

def fetchwmt13commoncrawl(m:Manager)->DRef:
  return fetchurl(m,
    name='training-parallel-commoncrawl',
    url='http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
    sha256='c7a74e2ea01ac6c920123108627e35278d4ccb5701e15428ffa34de86fa3a9e5')

def fetchwmt13europarl(m:Manager)->DRef:
  return fetchurl(m,
    name='training-parallel-europarl-v7',
    url='http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
    sha256='0224c7c710c8a063dfd893b0cc0830202d61f4c75c17eb8e31836103d27d96e7')

def fetchwmt17dev(m:Manager)->DRef:
  return fetchurl(m,
    name='newstest2013',
    url=f'http://data.statmt.org/wmt17/translation-task/dev.tgz',
    sha256='9d5ff04a28496b7796904ea65e50e3837bab14dbdca88b1e063105f17513dca9')

def fetchnewstest2014(m:Manager)->DRef:
  name='newstest2014'
  return fetchurl(m,
    name=name,
    url=f'https://storage.googleapis.com/tf-perf-public/official_transformer/test_data/{name}.tgz',
    sha256='1d07bf20db2f4607bb5f3c228e24c1fa17bdfb66d650cc5e713353606fde0800')


def fetchwmtpack(m:Manager)->DRef:
  return mknode(m,{
    'wmt17parallel':fetchwmt17parallel(m),
    'wmt13commoncrawl':fetchwmt13commoncrawl(m),
    'wmt13europarl':fetchwmt13europarl(m),
    'wmt17dev':fetchwmt17dev(m),
    'newstest2014':fetchnewstest2014(m),
    })


def catfiles(m:Manager, files:List[RefPath], outname:Optional[str]=None)->DRef:
  """ Concatenate `files` into a single file named `outname` (defaults to
  'result') """
  outname_='result' if outname is None else outname
  def _realize(b:Build)->None:
    o=build_outpath(b)
    with open(join(o,outname_),'w') as dst:
      for rp in files:
        path=build_path(b,rp)
        nwritten=0
        with open(path,'r',newline='\n') as tgt:
          for line in tgt:
            dst.write(line.strip())
            dst.write('\n')
            nwritten+=1
        print(f'Written {nwritten} lines from {rp}')

  return mkdrv(m,mkconfig({'version':3, 'files':files, 'output':outname_}),
                 match_only(),
                 build_wrapper(_realize))

def trainfiles(m:Manager, lang1:str, lang2:str, europarl:Optional[bool]=None)->DRef:
  suffix=f"{lang2}-{lang1}"
  inputsname='inputs'
  europarl_= europarl if europarl is not None else ('ru' not in [lang1,lang2])

  inputs=catfiles(m, outname=inputsname, files=[
      [fetchwmt17parallel(m),'training',f'news-commentary-v12.{suffix}.{lang1}'],
      [fetchwmt13commoncrawl(m),'training-parallel-commoncrawl',f'commoncrawl.{suffix}.{lang1}'],
    ] + (
      [[fetchwmt13europarl(m),'training',f'europarl-v7.{suffix}.{lang1}']] \
          if europarl_ else []))

  targetsname='targets'
  targets=catfiles(m, outname=targetsname, files=[
      [fetchwmt17parallel(m),'training',f'news-commentary-v12.{suffix}.{lang2}'],
      [fetchwmt13commoncrawl(m),'training-parallel-commoncrawl',f'commoncrawl.{suffix}.{lang2}'],
    ] + (
      [[fetchwmt13europarl(m),'training',f'europarl-v7.{suffix}.{lang2}']] \
          if europarl_ else []))

  return mknode(m, {'name':'trainfiles',
                    'train_input_combined':[inputs,inputsname],
                    'train_target_combined':[targets,targetsname]})

def evalfiles(m:Manager, lang1:str, lang2:str)->DRef:
  inputsname='inputs'
  inputs=catfiles(m, files=[ [fetchwmt17dev(m),'dev',f'newstest2013.{lang1}'] ], outname=inputsname)
  targetsname='targets'
  targets=catfiles(m, files=[ [fetchwmt17dev(m),'dev',f'newstest2013.{lang2}'] ], outname=targetsname)
  return mknode(m, {'name':'evalfiles',
                    'eval_input_combined':[inputs,inputsname],
                    'eval_target_combined':[targets,targetsname]})



def wmttfrecs_(m:Manager, trainfiles:DRef, evalfiles:DRef)->WmtTfrecs:

  def _config():
    name = "wmttfrecs"
    train_tag = "train"
    train_shards = 100
    eval_tag = "eval"

    # Link to the raw train data collection
    train_input_combined = store_cattrs(trainfiles).train_input_combined
    train_target_combined = store_cattrs(trainfiles).train_target_combined

    # Link to the raw eval data collection
    eval_input_combined = store_cattrs(evalfiles).eval_input_combined
    eval_target_combined = store_cattrs(evalfiles).eval_target_combined

    # Desired number of subtokens in the vocabulary list.
    target_vocab_size = 32768
    # Accept vocabulary if size is within this threshold.
    target_threshold = 327
    # Minimum length of subtoken to pass the subtoken filter
    train_data_min_count:Optional[int] = 6
    # Vocabulry filename
    vocab_file = "vocab.%d" % target_vocab_size
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
        vocab_file=join(o,c.vocab_file),
        files=train_combined,
        target_vocab_size=c.target_vocab_size,
        threshold=c.target_threshold,
        min_count=c.train_data_min_count)
    print('Encoding train files')
    encode_and_save_files(subtokenizer, o, train_combined, c.train_tag, c.train_shards)
    print('Encoding eval files')
    encode_and_save_files(subtokenizer, o, eval_combined, c.eval_tag, 1)

  return WmtTfrecs(mkdrv(m, _config(), match_only(), build_wrapper(_realize)))


def wmttfrecs(m:Manager, lang1:str, lang2:str)->WmtTfrecs:
  return wmttfrecs_(m,trainfiles(m,lang1,lang2),
                      evalfiles(m,lang1,lang2))

