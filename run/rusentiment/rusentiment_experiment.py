from pylightnix import ( RRef, Build, rref2path, rref2dref, match_some,
    realizeMany, match_latest, store_buildtime, store_buildelta, store_context,
    BuildArgs, mkdrv, build_wrapper, match_only, build_setoutpaths, readjson )

from stagedml.stages.all import *
from stagedml.stages.bert_finetune_glue import ( Model as BertClsModel,
    build as bert_finetune_build )
from stagedml.types import ( Dict, Union, Optional, List, Any )
from stagedml.core import ( protocol_rref_metric )
from stagedml.imports import ( FullTokenizer, MakeNdarray, EventAccumulator,
    STORE_EVERYTHING_SIZE_GUIDANCE, ScalarEvent, TensorEvent, Features, Feature,
    Example, Dataset, OrderedDict, read_csv, DataFrame, makedirs, json_dump )

from official.nlp.bert.classifier_data_lib import ( InputExample, InputFeatures,
    convert_single_example )

import numpy as np
import tensorflow as tf

class Runner:
  def __init__(self, rref:RRef):
    self.rref=rref
    self.tokenizer=FullTokenizer(
      vocab_file=mklens(rref).tfrecs.bert_vocab.syspath,
      do_lower_case=mklens(rref).tfrecs.lower_case.val)
    self.max_seq_length=mklens(rref).max_seq_length.val
    self.model=BertClsModel(
      BuildArgs(rref2dref(rref), store_context(rref), None, {}, {}))
    bert_finetune_build(self.model)
    self.model.model.load_weights(mklens(rref).checkpoint_full.syspath)

  def eval(self, sentences:List[str], batch_size:int=10)->List[List[float]]:

    @tf.function
    def _tf_step(inputs):
      return self.model.model_eval(inputs, training=False)

    def _step(inputs:List[Tuple[Any,Any,Any]]):
      return _tf_step((tf.constant([i[0] for i in inputs], dtype=tf.int64),
                       tf.constant([i[1] for i in inputs], dtype=tf.int64),
                       tf.constant([i[2] for i in inputs], dtype=tf.int64)))\
             .numpy().tolist()

    buf=[]
    outs:List[List[float]]=[]
    for i,s in enumerate(sentences):
      ie=InputExample(guid=f"eval_{i:04d}", text_a=s, label='dummy')
      f=convert_single_example(
        10, ie, ['dummy'], self.max_seq_length, self.tokenizer)
      buf.append((f.input_ids, f.input_mask, f.segment_ids))
      if len(buf)>=batch_size:
        outs.extend(_step(buf))
        buf=[]

    outs.extend(_step(buf))
    return outs


def rusentiment_process(m:Manager, model:DRef)->DRef:
  def _realize(b:Build):
    build_setoutpaths(b,1)
    r=Runner(mklens(b).model.rref)
    df=read_csv(mklens(b).model.tfrecs.refdataset.output_tests.syspath)
    labels=sorted(df['label'].value_counts().keys())
    df['pred']=[labels[np.argmax(probs)] for probs in r.eval(list(df['text']))]
    confusion={l:{l2:0.0 for l2 in labels} for l in labels}
    for i,row in df.iterrows():
      confusion[row['label']][row['pred']]+=1.0
    confusion={l:{l2:i/sum(items.values()) for l2,i in items.items()} \
               for l,items in confusion.items()}
    with open(mklens(b).confusion_matrix.syspath,'w') as f:
      json_dump(confusion,f,indent=4)
    df.to_csv(mklens(b).prediction.syspath)

  return mkdrv(m, matcher=match_only(), realizer=build_wrapper(_realize),
    config=mkconfig({
      'model':model,
      'confusion_matrix':[promise, 'confusion_matrix.json'],
      'prediction':[promise, 'prediction.csv'],
      'version':2,
      }))


def all_rusentiment_process(m):
  refbert=all_multibert_finetune_rusentiment(m)
  return rusentiment_process(m,refbert)


import altair as alt
from altair import Chart
from altair_saver import save as altair_save

def confusion_matrix(rref:RRef):
  data={'label':[],'pred':[],'val':[]}
  for l,items in readjson(mklens(rref).confusion_matrix.syspath).items():
    for l2,val in items.items():
      data['label'].append(l)
      data['pred'].append(l2)
      data['val'].append(val)
  altair_save(alt.Chart(DataFrame(data)).mark_rect().encode(
    x='label:O',y='pred:O',color='val:Q'), 'cm.png')


