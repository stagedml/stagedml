from pylightnix import ( Hash, RefPath, Build, Path, Config, Manager, RRef,
    DRef, Context, build_wrapper, build_path, build_outpath, build_cattrs,
    mkdrv, rref2path, mkbuild, mkconfig, match_only, instantiate, realize,
    lsref, catref, store_cattrs, dirhash, fetchlocal, mknode,
    mklens, instantiate, realize, repl_realize, repl_build, promise )

from stagedml.core import STAGEDML_ROOT
from stagedml.utils.files import ( system, flines, writelines, readlines )

from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile, read_csv, DataFrame, FullTokenizer,
    file_based_convert_examples_to_features, json_dump )

from stagedml.types import ( Optional, Any, List, Tuple, Union, Rusent, Set,
    Iterable, BertFinetuneTFR )

from stagedml.datasets.glue.processors import ( InputExample )

import pandas as pd


def fetchrusent(m:Manager, shuffle:bool=True)->Rusent:
  return Rusent(fetchlocal(m,
    name='fetchrusent',
    envname='STAGEDML_RUSENTIMENT',
    sha256='cbc02dfbfaee81eda1f192b5280f05fbda41fb1ab9952cb4d8f7b0ff227c968d',
    output_preselected=[promise, 'rusentiment.tar', 'rusentiment_preselected_posts.csv'],
    output_random=[promise, 'rusentiment.tar', 'rusentiment_random_posts.csv'],
    output_tests=[promise, 'rusentiment.tar', 'rusentiment_test.csv']))


def created_examples(df:DataFrame, type_hint:str)->Iterable[InputExample]:
  for i,row in df.iterrows():
    yield InputExample(guid=f"{type_hint}_{i:04d}", text_a=row.text, text_b=None,
      label=row.label)

def rusent_process(b:Build)->None:
  c=build_cattrs(b)
  o=build_outpath(b)

  df_presel=read_csv(mklens(b).refdataset.output_preselected.syspath)
  df_random=read_csv(mklens(b).refdataset.output_random.syspath)
  df_test=read_csv(mklens(b).refdataset.output_tests.syspath)

  df_train_valid=pd.concat([df_random,df_presel]).sample(frac=1).reset_index(drop=True)
  df_train_index=int(len(df_train_valid.index)*c.train_valid_ratio)
  df_train=df_train_valid.iloc[:df_train_index]
  df_valid=df_train_valid.iloc[df_train_index:]
  vocab_path=mklens(b).bert_vocab.syspath

  def _labels(df):
    return set(df['label'].value_counts().keys())

  labels=sorted(list(_labels(df_presel)|_labels(df_random)|_labels(df_test)))
  print(f"Labels are: {labels}")
  assert c.num_classes==len(labels)

  tokenizer = FullTokenizer(vocab_file=vocab_path, do_lower_case=c.lower_case)

  file_based_convert_examples_to_features(
    created_examples(df_train, 'train'),
    labels, c.max_seq_length, tokenizer, mklens(b).outputs.train.syspath)
  file_based_convert_examples_to_features(
    created_examples(df_valid, 'valid'),
    labels, c.max_seq_length, tokenizer, mklens(b).outputs.valid.syspath)
  file_based_convert_examples_to_features(
    created_examples(df_test, 'test'),
    labels, c.max_seq_length, tokenizer, mklens(b).outputs.test.syspath)


def rusent_tfrecords(m:Manager,
                     bert_vocab:RefPath,
                     lower_case:bool,
                     refdataset:Rusent)->BertFinetuneTFR:
  def _config():
    name='tfrecord-rusent'
    nonlocal bert_vocab
    nonlocal refdataset
    nonlocal lower_case
    task_name='rusentiment'
    outputs={'train':[promise,'train.tfrecord'],
             'valid':[promise,'valid.tfrecord'],
             'test':[promise,'test.tfrecord']}
    max_seq_length=128
    train_valid_ratio=0.9
    num_classes=5
    return locals()

  return BertFinetuneTFR(
    mkdrv(m,
      config=mkconfig(_config()),
      matcher=match_only(),
      realizer=build_wrapper(rusent_process)))

