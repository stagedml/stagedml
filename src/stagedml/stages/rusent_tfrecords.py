from stagedml.imports.sys import ( environ, join, basename, dedent,
                                  contextmanager, isfile, read_csv, DataFrame,
                                  OrderedDict, Hash, RefPath, Build, Path,
                                  Config, Manager, RRef, DRef, Context,
                                  build_wrapper, build_path, build_outpath,
                                  build_cattrs, mkdrv, rref2path, mkconfig,
                                  match_only, instantiate, realize, lsref,
                                  catref, store_cattrs, dirhash, fetchlocal,
                                  mknode, mklens, instantiate, realize,
                                  repl_realize, repl_build, promise )

from stagedml.imports.tf import ( TFRecordWriter, Example, Features, Feature,
                                 Int64List )

from stagedml.types import ( Optional, Any, List, Tuple, Union, Rusent, Set,
                            Iterable, BertFinetuneTFR )

from stagedml.datasets.glue.processors import (InputExample)

from keras_bert import Tokenizer, load_vocabulary

from stagedml.core import ( flines, readlines )

import pandas as pd


def create_examples(df:DataFrame, type_hint:str)->List[InputExample]:
  result=[]
  for i,row in df.iterrows():
    result.append(InputExample(guid=f"{type_hint}_{i:04d}", text_a=row.text,
                               text_b=None, label=row.label))
  return result

def make(b:Build)->None:
  c=build_cattrs(b)
  o=build_outpath(b)

  df_presel=read_csv(mklens(b).refdataset.output_preselected.syspath)
  df_random=read_csv(mklens(b).refdataset.output_random.syspath)
  df_test=read_csv(mklens(b).refdataset.output_tests.syspath)

  df_train_valid=pd.concat([df_random,df_presel]).sample(frac=1).reset_index(drop=True)
  df_train_index=int(len(df_train_valid.index)*c.train_valid_ratio)
  df_train=df_train_valid.iloc[:df_train_index]
  df_valid=df_train_valid.iloc[df_train_index:]

  vocab = load_vocabulary(mklens(b).bert_vocab.syspath)
  tokenizer = Tokenizer(vocab, cased=not mklens(b).lower_case.val)

  llabels = lambda df: set(df['label'].value_counts().keys())
  labels=sorted(list(llabels(df_presel) | llabels(df_random) | llabels(df_test)))
  assert c.num_classes==len(labels)
  label_ids = {label:i for i,label in enumerate(labels)}
  max_seq_length=mklens(b).max_seq_length.val

  def _tokenize(examples:List[InputExample], outfile):
    def _ints(values):
      return Feature(int64_list=Int64List(value=list(values)))
    with TFRecordWriter(outfile) as writer:
      for (i,e) in enumerate(examples):
        input_ids,segment_ids = tokenizer.encode(e.text_a, e.text_b,
                                                 max_seq_length)
        label_id = label_ids[e.label]

        features = OrderedDict()
        features["input_ids"] = _ints(input_ids)
        features["segment_ids"] = _ints(segment_ids)
        features["label_ids"] = _ints([label_id])
        te = Example(features=Features(feature=features))

        writer.write(te.SerializeToString())

  _tokenize(create_examples(df_train, 'train'), mklens(b).outputs.train.syspath)
  _tokenize(create_examples(df_valid, 'valid'), mklens(b).outputs.valid.syspath)
  _tokenize(create_examples(df_test, 'test'), mklens(b).outputs.test.syspath)

def tfrec_rusent(m:Manager,
                 bert_vocab:RefPath,
                 lower_case:bool,
                 refdataset:Rusent)->BertFinetuneTFR:
  def _config():
    name='tfrecord-rusent'
    nonlocal bert_vocab
    nonlocal refdataset
    nonlocal lower_case
    task_name = 'rusentiment'
    outputs = {'train':[promise,'train.tfrecord'],
               'valid':[promise,'valid.tfrecord'],
               'test':[promise,'test.tfrecord']}
    max_seq_length = 128
    train_valid_ratio = 0.9
    num_classes = 5
    return locals()

  return BertFinetuneTFR(
    mkdrv(m,
      config=mkconfig(_config()),
      matcher=match_only(),
      realizer=build_wrapper(make)))


