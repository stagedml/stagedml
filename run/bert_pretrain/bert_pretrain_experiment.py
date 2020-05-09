from stagedml.imports import makedirs
from stagedml.stages.all import *

def run(task_name:str='MRPC',
        epoches:int=200,
        epoches_step:int=20)->tuple:
  """ Finetune BERT on GLUE dataset """

  def _pretrain_stage(nepoch:int, resume_rref:Optional[RRef]):
    return partial(all_minibert_pretrain,
                   train_epoches=nepoch,
                   resume_rref=resume_rref)

  def _finetune_stage(nepoch:int)->Stage:
    def _stage(m)->BertGlue:
      refglue=all_fetchglue(m)
      refbert=_pretrain_stage(nepoch, None)(m)
      gluetfr=glue_tfrecords(m, task_name,
          bert_vocab=mklens(refbert).bert_vocab.refpath,
          lower_case=mklens(refbert).cased.val==False,
          refdataset=refglue)
      tfbert=bert_finetune_glue(m,refbert,gluetfr)
      return tfbert
    return _stage

  print('Begin pretraining')

  pretrained:Dict[int,RRef]={}
  finetuned:Dict[int,RRef]={}
  for e in range(epoches_step,epoches+epoches_step,epoches_step):
    print('Pre-training up to epoch', e)
    pretrained[e]=realize(instantiate(
        _pretrain_stage(e, pretrained.get(e-epoches_step))))
    linkrref(pretrained[e],['bert_pretrain',f'epoch-{e}'])
    print('Fine-tunining up to epoch', e)
    finetuned[e]=realize(instantiate(_finetune_stage(e)))
    linkrref(finetuned[e],['bert_pretrain',f'epoch-{e}'])
  return pretrained, finetuned




import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.framework.tensor_util import MakeNdarray
from tensorboard.backend.event_processing.event_accumulator import ScalarEvent

def tensorboard_tags(
    rref:RRef,
    subfolder:str='train'
    )->Dict[str,Union[list,bool]]:
  path=join(rref2path(rref),subfolder)
  event_acc = EventAccumulator(path, {
      'compressedHistograms': 10,
      'images': 0,
      'tensors':10,
      'scalars': 100,
      'histograms': 1
  })
  event_acc.Reload()
  return event_acc.Tags()


def tensorboard_scalar_events(
    rref:RRef, subfolder:str, tag:str)->ScalarEvent:
  path=join(rref2path(rref),subfolder)
  event_acc=EventAccumulator(path, {
      'compressedHistograms': 10, 'images': 0,
      'scalars': 100, 'histograms': 1 })
  event_acc.Reload()
  return event_acc.Scalars(tag)

import altair as alt
from altair import Chart
import pandas as pd
from pandas import DataFrame
from stagedml.imports import ( join, environ, makedirs, defaultdict )
from altair_saver import save as altair_save

def altair_print(chart:Chart, png_filename:str, alt:str='',
                 width:Optional[str]=None, maxwidth:Optional[str]=None)->None:
  out=environ['REPOUT']
  makedirs(out, exist_ok=True)
  altair_save(chart, join(out,png_filename))
  opts=' '.join([
      f"width={width}" if width else "",
      f"max-width={maxwidth}" if maxwidth else ""])
  print("![%s](%s){%s}"%(alt, join(out,png_filename), opts))

# import altair_viewer
# # FIXME a hack, see https://github.com/altair-viz/altair_viewer/issues/37
# v=altair_viewer.ChartViewer(False)
# altair_viewer.display = v.display
# altair_viewer.render = v.render
# altair_viewer.show = v.show
# import altair as alt
# import pandas as pd
# from pandas import DataFrame


