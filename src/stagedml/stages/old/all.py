
def all_minibert_finetune_glue(m:Manager, task_name:str='MRPC',
                               num_instances:int=1)->BertGlue:
  """ Finetune mini-BERT on GLUE dataset

  Ref. https://github.com/google-research/bert
  """
  refbert=all_fetchminibert(m)
  refglue=all_fetchglue(m)
  glueref=glue_tfrecords(m, task_name,
    bert_vocab=mklens(refbert).bert_vocab.refpath,
    lower_case=mklens(refbert).cased.val==False, refdataset=refglue)
  def _new(d):
    mklens(d).name.val+='-mini'
    mklens(d).train_batch_size.val=8
    mklens(d).test_batch_size.val=8
  return redefine(bert_finetune_glue,new_config=_new)\
    (m,refbert,glueref, num_instances=num_instances)

def all_bert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
  """ Finetune base-BERT on GLUE dataset

  Ref. https://github.com/google-research/bert
  """
  refbert=all_fetchbert(m)
  refglue=all_fetchglue(m)
  vocab=mklens(refbert).bert_vocab.refpath
  glueref=glue_tfrecords(m, task_name, bert_vocab=vocab,
    lower_case=mklens(refbert).cased.val==False, refdataset=refglue)
  return bert_finetune_glue(m,refbert,glueref)

def all_multibert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
  """ Finetune milti-lingual base-BERT on GLUE dataset

  Ref. https://github.com/google-research/bert/blob/master/multilingual.md
  """
  refbert=all_fetch_multibert(m)
  refglue=all_fetchglue(m)
  vocab=mklens(refbert).bert_vocab.refpath
  glueref=glue_tfrecords(m, task_name, bert_vocab=vocab,
    lower_case=mklens(refbert).cased.val==False, refdataset=refglue)
  return bert_finetune_glue(m,refbert,glueref)

def all_largebert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
  """ Finetune milti-lingual base-BERT on GLUE dataset

  Ref. https://github.com/google-research/bert/blob/master/multilingual.md
  """
  refbert=all_fetch_largebert(m)
  refglue=all_fetchglue(m)
  vocab=mklens(refbert).bert_vocab.refpath
  glueref=glue_tfrecords(m, task_name, bert_vocab=vocab,
    lower_case=mklens(refbert).cased.val==False, refdataset=refglue)
  # return bert_finetune_glue(m,refbert,glueref)
  def _new(d):
    mklens(d).train_batch_size.val=1
    mklens(d).test_batch_size.val=1
  return redefine(bert_finetune_glue,new_config=_new)\
    (m,refbert, glueref, num_instances=1)

# def all_bert_finetune_glue(m:Manager, task_name:str='MRPC')->BertGlue:
#   """ Finetune BERT on GLUE dataset """
#   refbert=all_fetchbert(m)
#   glueref=all_glue_tfrecords(m,task_name)
#   return bert_finetune_glue(m,refbert,glueref)


def all_multibert_finetune_rusentiment(m:Manager):
  refbert=all_fetch_multibert(m)
  refdata=all_fetchrusent(m)
  vocab=mklens(refbert).bert_vocab.refpath
  reftfr=rusent_tfrecords(m, bert_vocab=vocab,
    lower_case=mklens(refbert).cased.val==False, refdataset=refdata)
  return bert_finetune_glue(m,refbert,reftfr)

def all_bert_finetune_squad11(m:Manager)->BertSquad:
  """ Finetune BERT on Squad-1.1 dataset """
  squadref=all_squad11_tfrecords(m)
  return bert_finetune_squad11(m,squadref)

def all_transformer_wmtenru(m:Manager)->TransWmt:
  """ Train a Transformer model on WMT En->Ru translation task """
  return transformer_wmt(m, all_wmtsubtok_enru(m))

def all_transformer_wmtruen(m:Manager)->TransWmt:
  """ Train a Transformer model on WMT Ru->En translation task """
  return transformer_wmt(m, all_wmtsubtok_ruen(m))

def all_basebert_pretrain(m:Manager, **kwargs)->BertPretrain:
  tfr=all_enwiki_tfrecords(m)
  return basebert_pretrain_wiki(m, tfr, **kwargs)

def all_minibert_pretrain(m:Manager, **kwargs)->BertPretrain:
  tfr=all_enwiki_tfrecords(m)
  return minibert_pretrain_wiki(m, tfr, **kwargs)

