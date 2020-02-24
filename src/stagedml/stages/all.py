""" This file contains the collection of top-level stages.

Example IPython session:

```python
from stagedml.stages.all import *
store_initialize()
rref=realize(instantiate(all_bert_finetune_glue, 'MRPC'))
rref2path(rref)
```
"""

from pylightnix import ( Manager, mknode, fetchurl, instantiate, realize,
    rref2path, store_initialize, lsref, catref, repl_realize,
    repl_continueBuild, repl_build, repl_rref, repl_cancel )

from stagedml.stages.fetchglue import fetchglue
from stagedml.stages.glue_tfrecords import glue_tfrecords
from stagedml.stages.bert_finetune_glue import bert_finetune_glue
from stagedml.stages.fetchsquad import fetchsquad11
from stagedml.stages.squad_tfrecords import squad11_tfrecords
from stagedml.stages.bert_finetune_squad import bert_finetune_squad11
from stagedml.stages.nl2bash.all import nl2bash
from stagedml.stages.fetchwmt import wmttfrecs
from stagedml.stages.transformer_wmt import transformer_wmt
from stagedml.stages.convnn_mnist import fetchmnist, convnn_mnist

from stagedml.types import ( DRef, Glue, Squad11, GlueTFR, Squad11TFR, BertCP,
    BertGlue, BertSquad, NL2Bash, TransWmt, WmtTfrecs, ConvnnMnist )

all_fetchglue = fetchglue
all_fetchsquad11 = fetchsquad11

def all_fetchbert(m:Manager)->BertCP:
  return BertCP(fetchurl(m,
    name='uncased-bert',
    url='https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/uncased_L-12_H-768_A-12.tar.gz',
    sha256='018ef0ac65fc371f97c1e2b1ede59b5afb2d9e1da0217eb5072888940fb51978'))

def all_glue_tfrecords(m:Manager)->GlueTFR:
  refbert=all_fetchbert(m)
  refglue=all_fetchglue(m)
  return glue_tfrecords(m, dataset_name='glue', refbert=refbert, refdataset=refglue)

def all_squad11_tfrecords(m:Manager)->Squad11TFR:
  bertref=all_fetchbert(m)
  squadref=all_fetchsquad11(m)
  return squad11_tfrecords(m, bertref, squadref)

def all_bert_finetune_glue(m:Manager, task_name:str)->BertGlue:
  glueref=all_glue_tfrecords(m)
  return bert_finetune_glue(m,task_name,glueref)

def all_bert_finetune_squad11(m:Manager)->BertSquad:
  squadref=all_squad11_tfrecords(m)
  return bert_finetune_squad11(m,squadref)

def all_nl2bash(m:Manager)->NL2Bash:
  return nl2bash(m)

def all_wmttfrecs_enru(m:Manager)->WmtTfrecs:
  return wmttfrecs(m, 'en', 'ru')

def all_wmttfrecs_ende(m:Manager)->WmtTfrecs:
  return wmttfrecs(m, 'en', 'de')

def all_transformer_wmtenru(m:Manager)->TransWmt:
  return transformer_wmt(m, all_wmttfrecs_enru(m))

all_fetchmnist = fetchmnist

def all_convnn_mnist(m:Manager)->ConvnnMnist:
  return convnn_mnist(m, fetchmnist(m))

