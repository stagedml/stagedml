from pylightnix import ( RRef, rref2path, rref2dref, match_some, realizeMany,
    match_latest, store_buildtime, store_buildelta, store_context, BuildArgs )

from stagedml.imports import makedirs
from stagedml.stages.all import *
from stagedml.stages.bert_finetune_glue import ( Model as BertClsModel,
    build as bert_finetune_build )
from stagedml.types import ( Dict, Union, Optional, List, Any )
from stagedml.core import ( protocol_rref_metric )
from stagedml.imports import ( FullTokenizer, MakeNdarray, EventAccumulator,
    STORE_EVERYTHING_SIZE_GUIDANCE, ScalarEvent, TensorEvent, Features, Feature,
    Example, Dataset, OrderedDict )

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











