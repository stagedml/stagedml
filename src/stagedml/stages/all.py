""" This file contains the collection of top-level stages.

Example IPython session:

```python
from stagedml.stages.all import *
store_initialize()
rref=realize(instantiate(all_bert_finetune_glue, 'MRPC'))
rref2path(rref)
```
"""

from pylightnix import ( Path, RRef, Manager, mknode, fetchurl, instantiate,
    realize, rref2path, store_initialize, shell, lsref, catref, repl_realize,
    repl_continueBuild, repl_build, repl_rref, repl_cancelBuild, store_gc,
    rmref, mklens, promise, claim, path2rref, rref2path, store_dref2path,
    dirsize, store_config, config_name, redefine, mkconfig )

from stagedml.stages.fetchglue import fetchglue
from stagedml.stages.glue_tfrecords import glue_tfrecords
from stagedml.stages.bert_finetune_glue import bert_finetune_glue
from stagedml.stages.fetchsquad import fetchsquad11
from stagedml.stages.squad_tfrecords import squad11_tfrecords
from stagedml.stages.bert_finetune_squad import bert_finetune_squad11
# from stagedml.stages.nl2bash.all import nl2bash
from stagedml.stages.fetchnl2bash import fetchnl2bash, nl2bashSubtok
from stagedml.stages.fetchwmt import wmtsubtok, wmtsubtokInv
from stagedml.stages.transformer_wmt import transformer_wmt
# from stagedml.stages.transformer2 import transformer2
from stagedml.stages.convnn_mnist import fetchmnist, convnn_mnist
from stagedml.stages.fetchenwiki import fetchwiki, extractwiki
from stagedml.stages.bert_pretrain_wiki import ( bert_pretraining_tfrecords,
    bert_pretrain_wiki )

from stagedml.types import ( Set, Tuple, List, DRef, Glue, Squad11, GlueTFR,
    Squad11TFR, BertCP, BertGlue, BertSquad, NL2Bash, TransWmt, WmtSubtok,
    ConvnnMnist, Wikidump, Wikitext, WikiTFR, BertPretrain )
from stagedml.core import ( lrealize, tryrealize, STAGEDML_EXPERIMENTS,
    diskspace_h, linkrref, realize_epoches )
from stagedml.imports import ( walk, join, abspath, islink, partial )

from beautifultable import BeautifulTable

all_fetchglue = fetchglue
all_fetchsquad11 = fetchsquad11

def all_fetchbert(m:Manager)->BertCP:
  """ Fetch BERT-base pretrained checkpoint from the Google cloud """
  return BertCP(fetchurl(m,
    name='uncased-bert',
    url='https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/uncased_L-12_H-768_A-12.tar.gz',
    sha256='018ef0ac65fc371f97c1e2b1ede59b5afb2d9e1da0217eb5072888940fb51978',
    bert_config=[promise,'uncased_L-12_H-768_A-12','bert_config.json'],
    bert_vocab=[promise,'uncased_L-12_H-768_A-12','vocab.txt'],
    bert_ckpt=[claim,'uncased_L-12_H-768_A-12','bert_model.ckpt']
    ))

def all_glue_tfrecords(m:Manager, task_name:str)->GlueTFR:
  """ Fetch and preprocess GLUE dataset """
  refbert=all_fetchbert(m)
  refglue=all_fetchglue(m)
  return glue_tfrecords(m, task_name, refbert=refbert, refdataset=refglue)

def all_squad11_tfrecords(m:Manager)->Squad11TFR:
  """ Fetch and preprocess Squad-1.1 dataset """
  bertref=all_fetchbert(m)
  squadref=all_fetchsquad11(m)
  return squad11_tfrecords(m, bertref, squadref)

def all_bert_finetune_glue(m:Manager, task_name:str)->BertGlue:
  """ Finetune BERT on GLUE dataset """
  glueref=all_glue_tfrecords(m,task_name)
  return bert_finetune_glue(m,glueref)

def dryrun_bert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
  """ Train a simple convolutional model on MNIST """
  def _new_config(d):
    d['name']+='-dryrun'
    d['train_epoches']=1
    d['dataset_size']=100
    return mkconfig(d)
  return redefine(partial(all_bert_finetune_glue, task_name=task_name), new_config=_new_config)(m)

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

all_fetchmnist = fetchmnist

def all_convnn_mnist(m:Manager)->ConvnnMnist:
  """ Train a simple convolutional model on MNIST """
  return convnn_mnist(m, fetchmnist(m))

def dryrun_convnn_mnist(m:Manager)->ConvnnMnist:
  """ Dry-run a simple convolutional model on MNIST """
  def _new_config(d):
    d['name']+='-dryrun'
    d['num_epoches']=1
    return mkconfig(d)
  return redefine(all_convnn_mnist, new_config=_new_config)(m)

def all_fetchenwiki(m:Manager)->Wikitext:
  """ Fetch and extract english wikipedia dump """
  wikidump=fetchwiki(m, dumpname='enwiki',
                        dumpdate='20200301',
                        sha1='852dfec9eba3c4d5ec259e60dca233b6a777a05e')
  return extractwiki(m,wikidump)

def all_fetchruwiki(m:Manager)->Wikitext:
  """ Fetch and extract russian wikipedia dump.

  Ref. https://dumps.wikimedia.org/enwiki/20200301/dumpstatus.json
  """
  wikidump=fetchwiki(m, dumpname='ruwiki',
                        dumpdate='20200301',
                        sha1='9f522ccf2931497e99a12d001a3bc7910f275519')
  return extractwiki(m,wikidump)

def all_bert_pretraining_tfrecords(m:Manager)->WikiTFR:
  b=all_fetchbert(m)
  return bert_pretraining_tfrecords(m,
      vocab_file=mklens(b).bert_vocab.refpath,
      wiki=all_fetchenwiki(m))

def all_bert_pretrain(m:Manager, **kwargs)->BertPretrain:
  tfr=all_bert_pretraining_tfrecords(m)
  return bert_pretrain_wiki(m, tfr, **kwargs)

def dryrun_bert_pretrain(m:Manager, train_epoches=1, resume_rref=None)->ConvnnMnist:
  """ Dry-run a simple convolutional model on MNIST """
  def _new_config(d):
    d['name']+='-dryrun'
    d['train_steps_per_loop']=1
    d['train_steps_per_epoch']=10
    return mkconfig(d)
  return redefine(partial(all_bert_pretrain,
                          train_epoches=train_epoches,
                          resume_rref=resume_rref), new_config=_new_config)(m)



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

  for root, dirs, filenames in walk(STAGEDML_EXPERIMENTS, topdown=True):
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
    t=BeautifulTable()
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
    print(f"Run `gc(force=True)` to remove the above references and free {diskspace_h(total_freed)}.")

