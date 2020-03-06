from os import environ, makedirs
from os.path import join, isdir
from json import dump as json_dump
from typing import Optional,Any,List,Tuple,Union

from pylightnix import ( Manager, Config, build_cattrs, build_outpath,
    mkdrv, match_only, store_cattrs, mklens, promise, mkconfig )

from stagedml.utils.files import json_read
from stagedml.core import ( ProtocolBuild, protocol_add, protocolled )
from stagedml.types import BertCP, Squad11, Squad11TFR

from stagedml.datasets.squad.tfrecord import ( predict_squad,
    generate_tf_record_from_json_file )

def config(bertref:BertCP, squadref:Squad11)->Config:
  name = 'squad_tfrecords'
  bert_vocab = mklens(bertref).bert_vocab.refpath
  do_lower_case = 'uncased' in mklens(bertref).url.val

  train = mklens(squadref).train.refpath
  dev = mklens(squadref).dev.refpath

  output_train = [promise, 'train.tfrecord']
  output_eval = [promise, 'eval.tfrecord']
  output_meta = [promise, 'meta.json']

  max_seq_length=128
  max_query_length = 64
  fine_tuning_task_type='squad'
  doc_stride = 64
  version_2_with_negative = False
  eval_batch_size = 8
  config_version = 3
  return mkconfig(locals())

def process(b:ProtocolBuild)->None:
  c=build_cattrs(b)
  o=build_outpath(b)

  print('Processing train tfrecords')
  number_of_train_examples = \
      generate_tf_record_from_json_file(
        mklens(b).train.syspath,
        mklens(b).bert_vocab.syspath,
        join(o,'train.tfrecord'),
        c.max_seq_length,
        c.do_lower_case,
        c.max_query_length,
        c.doc_stride,
        c.version_2_with_negative)

  print('Processing eval tfrecords')
  number_of_eval_examples = \
      predict_squad(
        input_file_path=mklens(b).dev.syspath,
        output_file=join(o,'eval.tfrecord'),
        vocab_file=mklens(b).bert_vocab.syspath,
        doc_stride=c.doc_stride,
        predict_batch_size=c.eval_batch_size,
        max_query_length=c.max_query_length,
        max_seq_length=c.max_seq_length,
        do_lower_case=c.do_lower_case,
        version_2_with_negative=c.version_2_with_negative)


  with open(join(o,'meta.json'),'w') as f:
    json_dump({
        "task_type": "bert_squad",
        "train_data_size": number_of_train_examples,
        "eval_data_size": number_of_eval_examples,
        "max_seq_length": c.max_seq_length,
        "max_query_length": c.max_query_length,
        "doc_stride": c.doc_stride,
        "version_2_with_negative": c.version_2_with_negative,
    },f)

  protocol_add(b, 'process')
  return


def squad11_tfrecords(m:Manager, bertref:BertCP, squadref:Squad11)->Squad11TFR:
  return Squad11TFR(
    mkdrv(m,
      config=config(bertref, squadref),
      matcher=match_only(),
      realizer=protocolled(process)))


