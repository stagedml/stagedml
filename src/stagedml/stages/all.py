""" This file contains the collection of top-level stages.

Example IPython session:

```python
from stagedml.stages.all import *
store_initialize()
rref=realize(instantiate(all_bert_finetune_glue, 'MRPC'))
rref2path(rref)
```
"""

from pylightnix import ( RRef, Manager, mknode, fetchurl, instantiate, realize,
    rref2path, store_initialize, shell, lsref, catref, repl_realize,
    repl_continueBuild, repl_build, repl_rref, repl_cancelBuild, store_gc,
    rmref, mklens, promise, claim)

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
from stagedml.stages.bert_pretrain_wiki import bert_pretraining_tfrecords

from stagedml.types import ( Set, Tuple, List, DRef, Glue, Squad11, GlueTFR,
    Squad11TFR, BertCP, BertGlue, BertSquad, NL2Bash, TransWmt, WmtSubtok,
    ConvnnMnist, Wikidump, Wikitext, WikiTFR )
from stagedml.core import ( lrealize, tryrealize )

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

def all_fetchenwiki(m:Manager)->DRef:
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


def gc(force:bool=False)->None:
  """ Run the garbage collector. Pass `focrce=True` to actually delete
  the data """
  drefs,rrefs=store_gc(keep_drefs=[], keep_rrefs=\
    filter(lambda x: x is not None, [tryrealize(clo) for clo in [  # type:ignore
      instantiate(all_convnn_mnist),
      instantiate(all_transformer_nl2bash),
      instantiate(all_transformer_wmtenru),
      instantiate(all_bert_finetune_glue,'MRPC'),
      instantiate(all_bert_finetune_squad11)
      ]])
  )

  if force:
    for rref in rrefs:
      rmref(rref)
    for dref in drefs:
      rmref(dref)
  else:
    print('The following refs will be deleted:')
    print('\n'.join([str(r) for r in drefs]+[str(r) for r in rrefs]))
    print('Re-run `gc` with `force=True` to actually remove them')

