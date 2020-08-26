from stagedml.imports.sys import (Stage, DRef, RRef, Build, BuildArgs, mkdrv,
                                  build_wrapper, match_only, build_setoutpaths,
                                  mklens, rref2dref, store_context, realize,
                                  instantiate, mkconfig, Manager, redefine,
                                  promise,

                                  join, read_csv, DataFrame, makedirs,
                                  json_dump, environ, contextmanager, partial)

from stagedml.imports.tf import (MakeNdarray, ScalarEvent, TensorEvent,
                                 Features, Feature, Example, Dataset)

from stagedml.stages.bert_finetune_glue_zhg import (Model as BertClsModel,
                                                    build as bert_finetune_build)

from stagedml.stages.all import (all_finetune_multibert_rusent)

from stagedml.types import (Dict, Union, Optional, List, Any, Tuple)

from stagedml.utils.tf import (tensorboard_tags, tensorboard_tensors, te2float)

from keras_bert import Tokenizer, load_vocabulary

# from official.nlp.bert.classifier_data_lib import ( InputExample, InputFeatures,
#     convert_single_example )

from altair import Chart
from altair_saver import save as altair_save

import numpy as np
import tensorflow as tf


@contextmanager
def prepare_markdown_image(image_filename:str, alt:str='', attrs:str=''):
  genimgdir=environ['REPOUT']
  repimgdir=environ.get('REPIMG',genimgdir)
  makedirs(genimgdir, exist_ok=True)
  yield join(genimgdir,image_filename)
  print("![%s](%s){%s}"%(alt, join(repimgdir,image_filename), attrs))

def altair_print(chart:Chart, png_filename:str, alt:str='', attrs:str='')->None:
  with prepare_markdown_image(png_filename, alt, attrs) as path:
    altair_save(chart, path)


class Runner:
  def __init__(self, rref:RRef):
    self.rref=rref
    vocab = load_vocabulary(mklens(rref).tfrecs.bert_vocab.syspath)
    self.tokenizer = Tokenizer(vocab,
                               cased=not mklens(rref).tfrecs.lower_case.val)
    self.max_seq_length=mklens(rref).max_seq_length.val
    self.model=BertClsModel(
      BuildArgs(rref2dref(rref), store_context(rref), None, {}, {}))
    bert_finetune_build(self.model)
    self.model.model.load_weights(mklens(rref).checkpoint_full.syspath)

  def eval(self, sentences:List[str], batch_size:int=10)->List[List[float]]:

    @tf.function
    def _tf_step(inputs):
      return self.model.model_eval(inputs, training=False)

    def _step(inputs:List[Tuple[Any,Any]]):
      return _tf_step((tf.constant([i[0] for i in inputs], dtype=tf.int64),
                       tf.constant([i[1] for i in inputs], dtype=tf.int64)))\
             .numpy().tolist()

    buf=[]
    outs:List[List[float]]=[]
    for i,s in enumerate(sentences):

      input_ids, segment_ids = \
        self.tokenizer.encode(first=s, second=None,
                              max_len=self.max_seq_length)
      buf.append((input_ids, segment_ids))
      if len(buf)>=batch_size:
        outs.extend(_step(buf))
        buf=[]

    outs.extend(_step(buf))
    return outs

# runner ends

learning_rates=[2e-5, 5e-5, 1e-4]

def all_multibert_finetune_rusentiment1(m:Manager, lr:Optional[float]=None):
  lr_ = lr if lr is not None else learning_rates[0]
  def _nc(c):
    mklens(c).name.val='rusent-pretrained'
    mklens(c).train_batch_size.val=8
    mklens(c).train_epoches.val=5
    mklens(c).lr.val=lr_
  return redefine(all_finetune_multibert_rusent, new_config=_nc)(m) # end1


def all_multibert_finetune_rusentiment0(m:Manager):
  def _nc(c):
    mklens(c).name.val='rusent-random'
    mklens(c).bert_ckpt_in.val=None
  return redefine(all_multibert_finetune_rusentiment1, new_config=_nc)(m) # end0


def bert_rusentiment_evaluation(m:Manager, stage:Stage)->DRef:

  def _realize(b:Build):
    build_setoutpaths(b,1)
    r=Runner(mklens(b).model.rref)
    df=read_csv(mklens(b).model.tfrecs.refdataset.output_tests.syspath)
    labels=sorted(df['label'].value_counts().keys())
    df['pred']=[labels[np.argmax(probs)] for probs in r.eval(list(df['text']))]
    confusion={l:{l2:0.0 for l2 in labels} for l in labels}
    for i,row in df.iterrows():
      confusion[row['label']][row['pred']]+=1.0
    confusion={l:{l2:i/sum(items.values()) for l2,i in items.items()}
               for l,items in confusion.items()}
    with open(mklens(b).confusion_matrix.syspath,'w') as f:
      json_dump(confusion,f,indent=4)
    df.to_csv(mklens(b).prediction.syspath)

  return mkdrv(m, matcher=match_only(), realizer=build_wrapper(_realize),
    config=mkconfig({
      'name':'confusion_matrix',
      'model':stage(m),
      'confusion_matrix':[promise, 'confusion_matrix.json'],
      'prediction':[promise, 'prediction.csv'],
      'version':2,
      }))

# eval ends

if __name__== '__main__':
  for lr in learning_rates:
    print(realize(instantiate(all_multibert_finetune_rusentiment1, lr=lr)))
  print(realize(instantiate(all_multibert_finetune_rusentiment0)))
  stage=partial(all_multibert_finetune_rusentiment1, lr=min(learning_rates))
  print(realize(instantiate(bert_rusentiment_evaluation, stage)))


