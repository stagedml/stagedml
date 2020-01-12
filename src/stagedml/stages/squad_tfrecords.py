from os import environ, makedirs
from os.path import join, isdir
from json import dump as json_dump
from typing import Optional,Any,List,Tuple,Union

from pylightnix import ( Model, Config, State, Hash, model_save,
                         protocol_add, model_config_ro, model_outpath,
                         state_add, search, state, Ref, store_refpath as refpath,
                         store_systempath, config_deref_ro )

from stagedml.utils.files import json_read
from stagedml.utils.tf import best, memlimit
from stagedml.utils.refs import BertCP, Squad11, Squad11TFR
from stagedml.utils.instantiate import Options, instantiate
from stagedml.datasets.squad.create_tfrecord import ( predict_squad,
                                                      generate_tf_record_from_json_file )

def config(bertref:BertCP, squadref:Squad11)->Config:
  name = 'squad_tfrecords'
  if isdir(store_systempath(refpath(bertref, ['uncased_L-12_H-768_A-12']))):
    vocab_refpath = refpath(bertref, ['uncased_L-12_H-768_A-12', 'vocab.txt'])
    bert_ckpt_refpath = refpath(bertref, ['uncased_L-12_H-768_A-12', 'bert_model.ckpt'])
    bert_config = json_read(store_systempath(refpath(bertref, ['uncased_L-12_H-768_A-12', 'bert_config.json'])))
    do_lower_case = True
  elif isdir(store_systempath(refpath(bertref, ['cased_L-12_H-768_A-12']))):
    vocab_refpath = refpath(bertref, ['cased_L-12_H-768_A-12', 'vocab.txt'])
    bert_ckpt_refpath = refpath(bertref, ['cased_L-12_H-768_A-12', 'bert_model.ckpt'])
    bert_config = json_read(store_systempath(refpath(bertref, ['cased_L-12_H-768_A-12', 'bert_config.json'])))
    do_lower_case = False
  else:
    raise ValueError(f"Unsupported BERT package format, please check {bertref}")
  train_refpath = config_deref_ro(squadref).train_refpath
  dev_refpath = config_deref_ro(squadref).dev_refpath
  max_seq_length=128
  max_query_length = 64
  fine_tuning_task_type='squad'
  doc_stride = 64
  version_2_with_negative = False
  eval_batch_size = 8
  config_version = 3
  return Config(locals())


def processed(s:State)->State:
  return state_add(s, 'process')
def process(m:Model)->Model:
  c=model_config_ro(m)
  o=model_outpath(m)

  number_of_train_examples = \
      generate_tf_record_from_json_file(
        store_systempath(c.train_refpath),
        store_systempath(c.vocab_refpath),
        join(o,'train.tfrecord'),
        c.max_seq_length,
        c.do_lower_case,
        c.max_query_length,
        c.doc_stride,
        c.version_2_with_negative)

  number_of_eval_examples = \
      predict_squad(
        input_file_path=store_systempath(c.dev_refpath),
        output_file=join(o,'eval.tfrecord'),
        vocab_file=store_systempath(c.vocab_refpath),
        doc_stride=c.doc_stride,
        predict_batch_size=c.eval_batch_size,
        max_query_length=c.max_query_length,
        max_seq_length=c.max_seq_length,
        do_lower_case=c.do_lower_case,
        version_2_with_negative=c.version_2_with_negative)


  with open(o+'/meta.json','w') as f:
    json_dump({
        "task_type": "bert_squad",
        "train_data_size": number_of_train_examples,
        "eval_data_size": number_of_eval_examples,
        "max_seq_length": c.max_seq_length,
        "max_query_length": c.max_query_length,
        "doc_stride": c.doc_stride,
        "version_2_with_negative": c.version_2_with_negative,
    },f)

  protocol_add(m, 'process')
  return m


def squad11_tfrecords(o:Options, bertref:BertCP, squadref:Squad11)->Squad11TFR:
  c=config(bertref, squadref)
  def _search():
    return search(processed(state(c)))
  def _build():
    return model_save(process(Model(c)))
  return  Squad11TFR(instantiate(o, _search, _build))

