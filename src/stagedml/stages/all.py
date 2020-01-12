""" Collection of well-known toplevel rules for training """
from pylightnix import mknode
from stagedml.utils.instantiate import Options
from stagedml.stages.fetchurl import fetchurl
from stagedml.stages.fetchglue import fetchglue
from stagedml.stages.fetchsquad import fetchsquad11
from stagedml.stages.glue_tfrecords import glue_tfrecords
from stagedml.stages.squad_tfrecords import squad11_tfrecords
# from stagedml.stages.bert_finetune_glue import bert_finetune_glue
# from stagedml.stages.bert_finetune_squad import bert_finetune_squad11

from stagedml.utils.refs import ( Ref, Glue, Squad11, GlueTFR, Squad11TFR, BertCP )

common_fetchglue = fetchglue
common_fetchsquad11 = fetchsquad11

def common_fetchbert(o:Options)->BertCP:
  return BertCP(fetchurl(o,
    name='uncased-bert',
    url='https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/uncased_L-12_H-768_A-12.tar.gz',
    sha256='018ef0ac65fc371f97c1e2b1ede59b5afb2d9e1da0217eb5072888940fb51978'))

def common_glue_tfrecords(o:Options)->GlueTFR:
  refbert=common_fetchbert(o)
  refglue=common_fetchglue(o)
  return glue_tfrecords(o, dataset_name='glue', refbert=refbert, refdataset=refglue)

def common_squad11_tfrecords(o:Options)->Squad11TFR:
  bertref=common_fetchbert(o)
  squadref=common_fetchsquad11(o)
  return squad11_tfrecords(o, bertref, squadref)

# def common_bert_finetune_glue(o:Options, task_name:str)->Ref:
#   glueref=common_glue_tfrecords(o)
#   return bert_finetune_glue(o,task_name,glueref)

# def common_bert_finetune_squad11(o:Options)->Ref:
#   squadref=common_squad11_tfrecords(o)
#   return bert_finetune_squad11(o,squadref)

