from stagedml.imports import makedirs
from stagedml.stages.all import *

def run(task_name:str='MRPC', epoches:int=200, epoches_step:int=20)->tuple:
  """ Finetune BERT on GLUE dataset """

  def _pretrain_stage(nepoch:int, resume_rref:Optional[RRef]):
    return partial(all_minibert_pretrain, train_epoches=nepoch, resume_rref=resume_rref)

  def _finetune_stage(nepoch:int)->Stage:
    def _stage(m)->BertGlue:
      refglue=all_fetchglue(m)
      refbert=_pretrain_stage(nepoch, None)(m)
      gluetfr=glue_tfrecords(m, task_name,
          bert_vocab=mklens(refbert).bert_vocab.refpath,
          refdataset=refglue)
      tfbert=bert_finetune_glue(m,refbert,gluetfr)
      return tfbert
    return _stage

  print('Begin pretraining')

  pretrained:Dict[int,RRef]={}
  finetuned:Dict[int,RRef]={}
  for e in range(epoches_step,epoches+epoches_step,epoches_step):
    out=Path(join(STAGEDML_EXPERIMENTS,'bert_pretrain',f'epoch-{e}'))
    makedirs(out, exist_ok=True)
    print('Pre-training up to epoch', e)
    pretrained[e]=realize(instantiate(_pretrain_stage(e, pretrained.get(e-epoches_step))))
    linkrref(pretrained[e],out)
    print('Fine-tunining up to epoch', e)
    finetuned[e]=realize(instantiate(_finetune_stage(e)))
    linkrref(finetuned[e],out)
  return pretrained, finetuned




import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.framework.tensor_util import MakeNdarray

def tensorboard_tags(path:str):
  event_acc = EventAccumulator(path, {
      'compressedHistograms': 10,
      'images': 0,
      'tensors':10,
      'scalars': 100,
      'histograms': 1
  })
  event_acc.Reload()
  return event_acc.Tags()


def tensorboard_scalar_events(path:str, tag:str):
  event_acc = EventAccumulator(path, {
      'compressedHistograms': 10,
      'images': 0,
      'scalars': 100,
      'histograms': 1
  })
  event_acc.Reload()
  return event_acc.Scalars(tag)


