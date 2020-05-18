from stagedml.stages.all import *



def dryrun_bert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
  """ Train a simple convolutional model on MNIST """
  def _new_config(d):
    mklens(d).name.val+='-dryrun'
    mklens(d).train_epoches.val=1
    mklens(d).dataset_size.val=100
  return redefine(stage=partial(all_bert_finetune_glue, task_name=task_name),
                  new_config=_new_config)(m)


def dryrun_convnn_mnist(m:Manager)->ConvnnMnist:
  """ Dry-run a simple convolutional model on MNIST """
  def _new_config(d):
    mklens(d).name.val+='-dryrun'
    mklens(d).num_epoches.val=1
  return redefine(all_convnn_mnist, new_config=_new_config)(m)


def dryrun_minibert_pretrain(m:Manager, train_epoches=3, resume_rref=None
                            )->BertPretrain:
  """ Dry-run a simple convolutional model on MNIST """
  def _nc1(c):
    mklens(c).dupe_factor.val=2
  tfrecs=redefine(all_enwiki_tfrecords, new_config=_nc1)(m)
  def _nc2(d):
    mklens(d).name.val+='-dryrun'
    mklens(d).train_steps_per_loop.val=1
    mklens(d).train_steps_per_epoch.val=30
  return redefine(partial(minibert_pretrain_wiki,
                          tfrecs=tfrecs,
                          train_epoches=train_epoches,
                          resume_rref=resume_rref), new_config=_nc2)(m)

