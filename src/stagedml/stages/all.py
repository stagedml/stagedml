""" This file contains the collection of top-level stages.

Example IPython session:

```python
from stagedml.stages.all import *
store_initialize()
rref=realize(instantiate(all_bert_finetune_glue, 'MRPC'))
rref2path(rref)
```
"""
from ipdb import set_trace

from pylightnix import ( Stage, Path, RRef, Manager, mknode, fetchurl,
    instantiate, realize, rref2path, store_initialize, shell, lsref, catref,
    repl_realize, repl_continueBuild, repl_build, repl_rref, repl_cancelBuild,
    store_gc, rmref, mklens, promise, claim, path2rref, rref2path,
    store_dref2path, dirsize, store_config, config_name, redefine, mkconfig,
    fetchlocal )

from stagedml.core import ( lrealize, tryrealize, diskspace_h, linkrref,
    linkrrefs, realize_recursive, depgraph, initialize, borrow, stub_exception )

from stagedml.types import ( Dict, Set, Tuple, List, Optional, Union, DRef,
    Glue, Squad11, GlueTFR, Squad11TFR, BertCP, BertGlue, BertSquad, NL2Bash,
    TransWmt, WmtSubtok, ConvnnMnist, Wikidump, Wikitext, WikiTFR, BertPretrain,
    BertFinetuneTFR, Any, Mnist, Rusent )

from stagedml.imports.sys import ( walk, join, abspath, islink, partial,
    get_terminal_size, BeautifulTable )

from stagedml.stages.fetchglue import fetchglue
from stagedml.stages.fetchsquad import fetchsquad11
from stagedml.stages.fetchwiki import fetchwiki, extractwiki, wikistat
from logging import getLogger

logger=getLogger(__name__)
error=logger.error

# def try_import(module:str, name:str)->Any:
#   """ Helper function which tries to import `name` from `module`, but set it to
#   the stub name in case of failure. Unfrotunately, this function can't preserve
#   types, so we avoid using it. """
#   try:
#     exec(f"from {module} import {name} as __x__")
#     return __x__ # type:ignore
#   except ModuleNotFoundError as e:
#     return partial(stub_exception, exception=e)

try:
  from stagedml.stages.glue_tfrecords import glue_tfrecords, glue_tasks
  from stagedml.stages.bert_finetune_glue import bert_finetune_glue
  from stagedml.stages.squad_tfrecords import squad11_tfrecords
  from stagedml.stages.bert_finetune_squad import bert_finetune_squad11
  # from stagedml.stages.nl2bash.all import nl2bash
  from stagedml.stages.fetchnl2bash import fetchnl2bash, nl2bashSubtok
  from stagedml.stages.fetchwmt import wmtsubtok, wmtsubtokInv
  from stagedml.stages.transformer_wmt import transformer_wmt
  # from stagedml.stages.transformer2 import transformer2
  from stagedml.stages.convnn_mnist import convnn_mnist
  from stagedml.stages.bert_pretrain_wiki import ( bert_pretrain_tfrecords,
      basebert_pretrain_wiki, minibert_pretrain_wiki )
  from stagedml.stages.rusent_tfrecords import ( rusent_tfrecords )
  from stagedml.utils.tf import ( runtb )
except ModuleNotFoundError as e:
  error("Some stages have failed to load and were replaced with stubs!")
  glue_tfrecords =          stub_exception(e) # type:ignore
  glue_tasks =              stub_exception(e) # type:ignore
  bert_finetune_glue =      stub_exception(e) # type:ignore
  bert_finetune_squad11 =   stub_exception(e) # type:ignore
  fetchnl2bash =            stub_exception(e) # type:ignore
  nl2bashSubtok =           stub_exception(e) # type:ignore
  wmtsubtok =               stub_exception(e) # type:ignore
  wmtsubtokInv =            stub_exception(e) # type:ignore
  transformer_wmt =         stub_exception(e) # type:ignore
  convnn_mnist =            stub_exception(e) # type:ignore
  bert_pretrain_tfrecords = stub_exception(e) # type:ignore
  basebert_pretrain_wiki =  stub_exception(e) # type:ignore
  minibert_pretrain_wiki =  stub_exception(e) # type:ignore
  rusent_tfrecords =        stub_exception(e) # type:ignore

#: Glue dataset
all_fetchglue = fetchglue

#: SQuad dataset
all_fetchsquad11 = fetchsquad11

def all_fetchmnist(m:Manager)->Mnist:
  return Mnist(
    fetchurl(m, name='mnist',
                mode='as-is',
                url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
                sha256='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'))

def all_fetchrusent(m:Manager, shuffle:bool=True)->Rusent:
  """ RuSentiment dataset:
  - [The Paper](https://www.aclweb.org/anthology/C18-1064.pdf)
  - [Annotation guidelines](https://github.com/text-machine-lab/rusentiment)
  """
  return Rusent(fetchlocal(m,
    name='fetchrusent',
    envname='STAGEDML_RUSENTIMENT',
    sha256='cbc02dfbfaee81eda1f192b5280f05fbda41fb1ab9952cb4d8f7b0ff227c968d',
    output_preselected=[promise, 'rusentiment.tar', 'rusentiment_preselected_posts.csv'],
    output_random=[promise, 'rusentiment.tar', 'rusentiment_random_posts.csv'],
    output_tests=[promise, 'rusentiment.tar', 'rusentiment_test.csv']))



def all_fetcholdbert(m:Manager)->BertCP:
  """ Fetch BERT-base pretrained checkpoint from the Google cloud """
  return BertCP(fetchurl(m,
    name='uncased-bert',
    url='https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/uncased_L-12_H-768_A-12.tar.gz',
    sha256='018ef0ac65fc371f97c1e2b1ede59b5afb2d9e1da0217eb5072888940fb51978',
    bert_config=[promise,'uncased_L-12_H-768_A-12','bert_config.json'],
    bert_vocab=[promise,'uncased_L-12_H-768_A-12','vocab.txt'],
    bert_ckpt=[claim,'uncased_L-12_H-768_A-12','bert_model.ckpt']
    ))

def all_fetchbert(m:Manager)->BertCP:
  """ Fetch BERT-base pretrained checkpoint from the Google cloud
  FIXME: rename to `all_fetch_basebert` """
  return BertCP(fetchurl(m,
    name='basebert-uncased',
    url='https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip',
    sha256='d15224e1e7d950fb9a8b29497ce962201dff7d27b379f5bfb4638b4a73540a04',
    bert_config=[promise,'uncased_L-12_H-768_A-12','bert_config.json'],
    bert_vocab=[promise,'uncased_L-12_H-768_A-12','vocab.txt'],
    bert_ckpt=[claim,'uncased_L-12_H-768_A-12','bert_model.ckpt'],
    cased=False
    ))

def all_fetch_multibert(m:Manager)->BertCP:
  """ Fetch BERT-base pretrained checkpoint from the Google cloud """
  return BertCP(fetchurl(m,
    name='basebert-multi-cased',
    url='https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip',
    sha256='60ec8d9a7c3cc1c15f6509a6cbe1a8d30b7f823b77cb7460f6d31383200aec9d',
    bert_config=[promise,'multi_cased_L-12_H-768_A-12','bert_config.json'],
    bert_vocab=[promise,'multi_cased_L-12_H-768_A-12','vocab.txt'],
    bert_ckpt=[claim,'multi_cased_L-12_H-768_A-12','bert_model.ckpt'],
    cased=True
    ))


def all_fetchminibert(m:Manager)->BertCP:
  """ FIXME: rename to `all_fetch_minibert` """
  return BertCP(fetchurl(m,
    name='minibert-uncased',
    url='https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip',
    sha256='5f087a0c6c73aed0b0a13f9a99dade56bece97d0594b713195821e031266fae9',
    bert_config=[promise,'uncased_L-4_H-256_A-4','bert_config.json'],
    bert_vocab=[promise,'uncased_L-4_H-256_A-4','vocab.txt'],
    bert_ckpt=[claim,'uncased_L-4_H-256_A-4','bert_model.ckpt'],
    cased=False
    ))


def all_fetch_largebert(m:Manager)->BertCP:
  """ FIXME: rename to `all_fetch_minibert` """
  folder="uncased_L-24_H-1024_A-16"
  return BertCP(fetchurl(m,
    name="largebert-uncased",
    url=f"https://storage.googleapis.com/bert_models/2018_10_18/{folder}.zip",
    sha256='f1999b33fa1f38ffe2d9b2871bdbb3d1ddf228f9077a70a7b2176b61cd46ddbc',
    bert_config=[promise,folder,'bert_config.json'],
    bert_vocab=[promise,folder,'vocab.txt'],
    bert_ckpt=[claim,folder,'bert_model.ckpt'],
    cased=False
    ))


def all_fetch_rubert(m:Manager):
  return BertCP(fetchurl(m,
    name='rubert-cased',
    url='http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz',
    sha256='c73145bfdbb91282fd12fe3943d2ddbbf86d1a0c6c810189cf8ec4ce13d6c0c1',
    bert_config=[promise,'rubert_cased_L-12_H-768_A-12_v2','bert_config.json'],
    bert_vocab=[promise,'rubert_cased_L-12_H-768_A-12_v2','vocab.txt'],
    bert_ckpt=[claim,'rubert_cased_L-12_H-768_A-12_v2','bert_model.ckpt'],
    cased=True
    ))

def all_glue_tfrecords(m:Manager, task_name:str, lower_case:bool)->GlueTFR:
  """ Fetch and preprocess GLUE dataset. `task_name` should be one of
  `glue_tasks()` """
  refbert=all_fetchbert(m)
  refglue=all_fetchglue(m)
  vocab=bert_vocab=mklens(refbert).bert_vocab.refpath
  return glue_tfrecords(m, task_name, bert_vocab=vocab,
    lower_case=lower_case, refdataset=refglue)

def all_squad11_tfrecords(m:Manager)->Squad11TFR:
  """ Fetch and preprocess Squad-1.1 dataset """
  bertref=all_fetchbert(m)
  squadref=all_fetchsquad11(m)
  return squad11_tfrecords(m, bertref, squadref)

def all_rusentiment_tfrecords(m:Manager)->BertFinetuneTFR:
  """ Fetch and preprocess RuSentiment dataset """
  bertref=all_fetchbert(m)
  rusentref=all_fetchrusent(m)
  return rusent_tfrecords(m, bert_vocab=mklens(bertref).bert_vocab.refpath,
                             lower_case=mklens(bertref).cased.val==False,
                             refdataset=rusentref)

def all_minibert_finetune_glue(m:Manager, task_name:str='MRPC',
                               num_instances:int=1)->BertGlue:
  """ Finetune mini-BERT on GLUE dataset

  Ref. https://github.com/google-research/bert
  """
  refbert=all_fetchminibert(m)
  refglue=all_fetchglue(m)
  glueref=glue_tfrecords(m, task_name,
    bert_vocab=mklens(refbert).bert_vocab.refpath,
    lower_case=mklens(refbert).cased.val==False, refdataset=refglue)
  def _new(d):
    mklens(d).name.val+='-mini'
    mklens(d).train_batch_size.val=8
    mklens(d).test_batch_size.val=8
  return redefine(bert_finetune_glue,new_config=_new)\
    (m,refbert,glueref, num_instances=num_instances)

def all_bert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
  """ Finetune base-BERT on GLUE dataset

  Ref. https://github.com/google-research/bert
  """
  refbert=all_fetchbert(m)
  refglue=all_fetchglue(m)
  vocab=mklens(refbert).bert_vocab.refpath
  glueref=glue_tfrecords(m, task_name, bert_vocab=vocab,
    lower_case=mklens(refbert).cased.val==False, refdataset=refglue)
  return bert_finetune_glue(m,refbert,glueref)

def all_multibert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
  """ Finetune milti-lingual base-BERT on GLUE dataset

  Ref. https://github.com/google-research/bert/blob/master/multilingual.md
  """
  refbert=all_fetch_multibert(m)
  refglue=all_fetchglue(m)
  vocab=mklens(refbert).bert_vocab.refpath
  glueref=glue_tfrecords(m, task_name, bert_vocab=vocab,
    lower_case=mklens(refbert).cased.val==False, refdataset=refglue)
  return bert_finetune_glue(m,refbert,glueref)

def all_largebert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
  """ Finetune milti-lingual base-BERT on GLUE dataset

  Ref. https://github.com/google-research/bert/blob/master/multilingual.md
  """
  refbert=all_fetch_largebert(m)
  refglue=all_fetchglue(m)
  vocab=mklens(refbert).bert_vocab.refpath
  glueref=glue_tfrecords(m, task_name, bert_vocab=vocab,
    lower_case=mklens(refbert).cased.val==False, refdataset=refglue)
  # return bert_finetune_glue(m,refbert,glueref)
  def _new(d):
    mklens(d).train_batch_size.val=1
    mklens(d).test_batch_size.val=1
  return redefine(bert_finetune_glue,new_config=_new)\
    (m,refbert, glueref, num_instances=1)

# def all_bert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
#   """ Finetune BERT on GLUE dataset """
#   refbert=all_fetchbert(m)
#   glueref=all_glue_tfrecords(m,task_name)
#   return bert_finetune_glue(m,refbert,glueref)


def all_multibert_finetune_rusentiment(m:Manager):
  refbert=all_fetch_multibert(m)
  refdata=all_fetchrusent(m)
  vocab=mklens(refbert).bert_vocab.refpath
  reftfr=rusent_tfrecords(m, bert_vocab=vocab,
    lower_case=mklens(refbert).cased.val==False, refdataset=refdata)
  return bert_finetune_glue(m,refbert,reftfr)

def all_bert_finetune_squad11(m:Manager)->BertSquad:
  """ Finetune BERT on Squad-1.1 dataset """
  squadref=all_squad11_tfrecords(m)
  return bert_finetune_squad11(m,squadref)

# def all_nl2bash(m:Manager)->NL2Bash:
#   return nl2bash(m)

def all_fetchnl2bash(m:Manager)->DRef:
  """ Fetch NL2BASH dataset """
  return fetchnl2bash(m)

def all_wmtsubtok_enru(m:Manager)->WmtSubtok:
  """ Subtokenize En->Ru WMT dataset """
  return wmtsubtok(m, 'en', 'ru')

def all_wmtsubtok_ruen(m:Manager)->WmtSubtok:
  """ Subtokenize Ru->En WMT dataset """
  return wmtsubtokInv(m, 'ru', 'en')

def all_wmtsubtok_ende(m:Manager)->WmtSubtok:
  """ Subtokenize En->De WMT dataset """
  return wmtsubtok(m, 'en', 'de')

def all_transformer_wmtenru(m:Manager)->TransWmt:
  """ Train a Transformer model on WMT En->Ru translation task """
  return transformer_wmt(m, all_wmtsubtok_enru(m))

def all_transformer_wmtruen(m:Manager)->TransWmt:
  """ Train a Transformer model on WMT Ru->En translation task """
  return transformer_wmt(m, all_wmtsubtok_ruen(m))

def all_nl2bashsubtok(m:Manager, **kwargs)->WmtSubtok:
  return nl2bashSubtok(m, **kwargs)

def all_transformer_nl2bash(m:Manager)->TransWmt:
  """ Train a Transformer model on NL2Bash dataset """
  return transformer_wmt(m, all_nl2bashsubtok(m))

def all_convnn_mnist(m:Manager)->ConvnnMnist:
  """ Train a simple convolutional model on MNIST """
  return convnn_mnist(m, all_fetchmnist(m))

def all_fetchenwiki(m:Manager)->Wikidump:
  """ Fetch English wikipedia dump """
  return fetchwiki(m, dumpname='enwiki',
                        dumpdate='20200301',
                        sha1='852dfec9eba3c4d5ec259e60dca233b6a777a05e')

def all_extractenwiki(m:Manager)->Wikitext:
  """ Extract English Wikipedia dump """
  return extractwiki(m,all_fetchenwiki(m))


def all_fetchruwiki(m:Manager)->Wikidump:
  """ Fetch and extract russian wikipedia dump.

  Ref. https://dumps.wikimedia.org/enwiki/20200301/dumpstatus.json
  """
  return fetchwiki(m, dumpname='ruwiki',
                        dumpdate='20200301',
                        sha1='9f522ccf2931497e99a12d001a3bc7910f275519')

def all_extractruwiki(m:Manager)->Wikitext:
  """ Extracts Russian Wikipedia dump """
  return extractwiki(m,all_fetchruwiki(m))

def all_enwiki_tfrecords(m:Manager)->WikiTFR:
  """ Create TFRecords dataset for English Wikipedia dump. Use vocabulary from
  Base BERT model by Google Research.
  """
  w=all_extractenwiki(m)
  b=all_fetchbert(m)
  return bert_pretrain_tfrecords(m,
      vocab_file=mklens(b).bert_vocab.refpath, wikiref=w)

def all_ruwiki_tfrecords(m:Manager)->WikiTFR:
  """ Create TFRecords dataset for Russian Wikipedia dump. Use vocabulary from
  Google's multilingual BERT model.
  """
  w=all_extractruwiki(m)
  b=all_fetch_multibert(m)
  return bert_pretrain_tfrecords(m,
    vocab_file=mklens(b).bert_vocab.refpath, wikiref=w)

def all_basebert_pretrain(m:Manager, **kwargs)->BertPretrain:
  tfr=all_enwiki_tfrecords(m)
  return basebert_pretrain_wiki(m, tfr, **kwargs)

def all_minibert_pretrain(m:Manager, **kwargs)->BertPretrain:
  tfr=all_enwiki_tfrecords(m)
  return minibert_pretrain_wiki(m, tfr, **kwargs)


def gcfind()->Tuple[Set[DRef],Set[RRef]]:
  """ Query the garbage collector. GC removes any model which is not under
  STAGEDML_EXPERIMENTS folder and is not in short list of pre-defined models.
  Return the links to be removed. Run `gc(force=True)` to actually remove the
  links.  """

  keep_rrefs=[x for x in
    (tryrealize(clo) for clo in [
      instantiate(all_convnn_mnist),
      instantiate(all_transformer_nl2bash),
      instantiate(all_transformer_wmtenru),
      instantiate(all_bert_finetune_glue,'MRPC'),
      instantiate(all_bert_finetune_squad11)
      ]) if x is not None]

  import stagedml.core
  for root, dirs, filenames in walk(stagedml.core.STAGEDML_EXPERIMENTS,
                                    topdown=True):
    for dirname in sorted(dirs):
      a=Path(abspath(join(root, dirname)))
      if islink(a):
        rref=path2rref(a)
        if rref is not None:
          keep_rrefs.append(rref)

  drefs,rrefs=store_gc(keep_drefs=[], keep_rrefs=keep_rrefs)
  return drefs,rrefs


def gc(force:bool=False):
  """ Run the garbage collector. GC removes any model which is not under
  STAGEDML_EXPERIMENTS folder and is not in short list of pre-defined models.

  Pass `focrce=True` to actually delete the data. """

  drefs,rrefs=gcfind()

  if force:
    for rref in rrefs:
      rmref(rref)
    for dref in drefs:
      rmref(dref)
  else:
    print('The following refs will be deleted:')
    t=BeautifulTable(max_width=get_terminal_size().columns)
    t.set_style(BeautifulTable.STYLE_MARKDOWN)
    t.width_exceed_policy = BeautifulTable.WEP_ELLIPSIS
    t.column_headers=['Name', 'RRef/DRef', 'Size']
    t.column_alignments['Name']=BeautifulTable.ALIGN_LEFT
    t.column_alignments['RRef/DRef']=BeautifulTable.ALIGN_LEFT
    d=sorted([(rref, dirsize(rref2path(rref))) for rref in rrefs] , key=lambda x:x[1])
    total_freed=0
    for rref,sz in d:
      t.append_row([config_name(store_config(rref)),rref,diskspace_h(sz)])
      total_freed+=sz
    print(t)
    print((f"Run `gc(force=True)` to remove the above references and free "
           f"{diskspace_h(total_freed)}."))

