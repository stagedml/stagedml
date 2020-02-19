from os import environ, makedirs
from os.path import join, isdir
from json import dump as json_dump
from typing import Optional,Any,List,Tuple,Union

from pylightnix import ( Manager, Config, build_cattrs, build_outpath,
    build_path, mkdrv, match_only, store_cattrs )

from stagedml.utils.files import json_read
from stagedml.utils.tf import ( ProtocolBuild, protocol_add, protocolled )
from stagedml.utils.refs import BertCP, Squad11, Squad11TFR

from stagedml.datasets.squad.tfrecord import ( predict_squad,
    generate_tf_record_from_json_file )

def config(bertref:BertCP, squadref:Squad11)->Config:
  name = 'squad_tfrecords'
  if 'uncased_L-12_H-768_A-12' in store_cattrs(bertref).url:
    vocab_refpath = [bertref, 'uncased_L-12_H-768_A-12', 'vocab.txt']
    bert_ckpt_refpath = [bertref, 'uncased_L-12_H-768_A-12', 'bert_model.ckpt']
    bert_config = [bertref, 'uncased_L-12_H-768_A-12', 'bert_config.json']
    do_lower_case = True
  elif 'cased_L-12_H-768_A-12' in store_cattrs(bertref).url:
    vocab_refpath = [bertref, 'cased_L-12_H-768_A-12', 'vocab.txt']
    bert_ckpt_refpath = [bertref, 'cased_L-12_H-768_A-12', 'bert_model.ckpt']
    bert_config = [bertref, 'cased_L-12_H-768_A-12', 'bert_config.json']
    do_lower_case = False
  else:
    raise ValueError(f"Unsupported BERT package format, please check {bertref}")
  train_refpath = store_cattrs(squadref).train_refpath
  dev_refpath = store_cattrs(squadref).dev_refpath
  max_seq_length=128
  max_query_length = 64
  fine_tuning_task_type='squad'
  doc_stride = 64
  version_2_with_negative = False
  eval_batch_size = 8
  config_version = 3
  return Config(locals())

def process(b:ProtocolBuild)->None:
  c=build_cattrs(b)
  o=build_outpath(b)

  number_of_train_examples = \
      generate_tf_record_from_json_file(
        build_path(b, c.train_refpath),
        build_path(b, c.vocab_refpath),
        join(o,'train.tfrecord'),
        c.max_seq_length,
        c.do_lower_case,
        c.max_query_length,
        c.doc_stride,
        c.version_2_with_negative)

  number_of_eval_examples = \
      predict_squad(
        input_file_path=build_path(b, c.dev_refpath),
        output_file=join(o,'eval.tfrecord'),
        vocab_file=build_path(b, c.vocab_refpath),
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


