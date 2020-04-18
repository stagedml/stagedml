from stagedml.stages.all import *

def run(task_name:str='MRPC')->dict:
  """ Finetune BERT on GLUE dataset """

  def _pretrained(nepoch):
    return partial(all_bert_pretrain, train_epoches=nepoch)

  def _run(m:Manager,e:int):
    refglue=all_fetchglue(m)
    refbert=_pretrained(e)(m)
    gluetfr=glue_tfrecords(m, task_name, bert_vocab=mklens(refbert).bert_vocab.refpath, refdataset=refglue)
    tfbert=bert_finetune_glue(m,refbert,gluetfr)
    return tfbert

  print('Begin pretraining')

  pretrains=realize_recursive(
      lambda e,rref: instantiate(_pretrained(e), resume_rref=rref),
      epoches=20, epoches_step=2)

  print('Begin fine-tuning on:', pretrains)
  res={}
  for e in pretrains.keys():
    print('Fine-tuning after epoch', e)
    res[e]=realize(instantiate(partial(_run,e=e)))
    out=Path(join(STAGEDML_EXPERIMENTS,'bert_pretrain',f'epoch-{e}'))
    linkrref(res[e],out)
  return res
