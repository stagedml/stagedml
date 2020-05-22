from pylightnix import ( build_setoutpaths, mkdrv, mkconfig, match_latest,
    build_wrapper, rref2dref, promise )
from stagedml.types import ( Callable )
from stagedml.imports import ( join, environ, makedirs, defaultdict,
    Dataset )
from stagedml.utils import ( EventAccumulator, MakeNdarray, ScalarEvent,
    dataset_iter_size, readstr, writestr, disable_gpu_prealloc,
    limit_gpu_memory, tensorboard_scalars, tensorboard_tags,
    tensorboard_tensors, te2float)
from stagedml.stages.bert_pretrain_wiki import bert_pretraining_dataset
from stagedml.stages.all import *

from altair import Chart
from pandas import DataFrame
from altair_saver import save as altair_save

import tensorflow as tf

def markdown_image(image_filename:str, alt:str='', attrs:str='')->tuple:
  genimgdir=environ['REPOUT']
  repimgdir=environ.get('REPIMG',genimgdir)
  makedirs(genimgdir, exist_ok=True)
  impath=join(genimgdir,image_filename)
  imtag="![%s](%s){%s}"%(alt, join(repimgdir,image_filename), attrs)
  return impath,imtag

def markdown_code(s:str)->str:
  return '`'+''.join([c if c!='`' else '\\'+c for c in s])+'`'

def markdown_url(url:str, descr=None)->str:
  return '['+(descr or url)+']('+url+')'

def markdown_altair(chart:Chart, png_filename:str, alt:str='', attrs:str='')->str:
  path,tag=markdown_image(png_filename, alt, attrs)
  altair_save(chart, path)
  return tag




disable_gpu_prealloc()

DEF_NEPOCHES:int=50
DEF_STEPS_PER_EPOCH:int=10000
DEF_EPOCHES_BETWEEN_FINETUNES:int=10
DEF_FINETUNE_TASK:str='MNLI-m'

# def ruwiki_fixup(c):
#   mklens(c).dupe_factor.val=4

def enwiki_fixup(c):
  mklens(c).dupe_factor.val=2

DatasetStage=Callable[[Manager],WikiTFR]

dataset_ru:DatasetStage=all_ruwiki_tfrecords
dataset_en:DatasetStage=redefine(all_enwiki_tfrecords, new_config=enwiki_fixup)


def experiment_dataset_ru()->RRef:
  return realize(instantiate(dataset_ru))

def experiment_dataset_en()->RRef:
  return realize(instantiate(dataset_en))

def num_tfrecords_approx_(ds):
  tfr:RRef=realize(instantiate(ds))
  input_pattern=f"{mklens(tfr).output.syspath}/*.tfrecord"
  dataset=Dataset.list_files(input_pattern, shuffle=False)
  input_files:List[str]=tf.io.gfile.glob(input_pattern)
  return len(input_files)*mklens(tfr).training_instances_per_output_file.val

num_tfrecords_approx_ru=partial(num_tfrecords_approx_, ds=dataset_ru)
num_tfrecords_approx_en=partial(num_tfrecords_approx_, ds=dataset_en)

def num_tfrecords_(ds, bs=100)->int:
  """ Return the number of input examples in the dataset_ru, `+-batch_size`
  FIXME: too slow """
  def _stage(m):
    wikiref=ds(m)
    config={'name':mklens(wikiref).name.val+'-size',
            'size':[promise, 'size.txt'],
            'wikiref':wikiref,
            'batch_size':bs}
    def _realize(b):
      build_setoutpaths(b,1)
      ds=bert_pretraining_dataset(mklens(b).wikiref.rref,bs,False,repeat=False)

      cnt=0
      for _ in ds:
        if cnt % 1000 == 0:
          print(cnt)
        cnt+=1
      size=cnt*bs
      # size=dataset_iter_size(lambda: ds)*bs
      writestr(mklens(b).size.syspath,str(size))
    return mkdrv(m, mkconfig(config),
                    match_latest(), build_wrapper(_realize))

  return int(readstr(mklens(realize(instantiate(_stage))).size.syspath))

num_tfrecords_en=partial(num_tfrecords_,ds=dataset_en)
num_tfrecords_ru=partial(num_tfrecords_,ds=dataset_ru)

def wikitext(dref):
  return mklens(dref).wikiref.dref

def num_words_(ds)->int:
  rref=realize(instantiate(lambda m: wikistat(m, wikitext(ds(m)))))
  return int(readstr(mklens(rref).output.wordnum.syspath))

num_words_en=partial(num_words_, ds=dataset_en)
num_words_ru=partial(num_words_, ds=dataset_ru)

def num_documents_(ds)->int:
  rref=realize(instantiate(lambda m: wikistat(m, wikitext(ds(m)))))
  return int(readstr(mklens(rref).output.docnum.syspath))

num_documents_en=partial(num_documents_,ds=dataset_en)
num_documents_ru=partial(num_documents_,ds=dataset_ru)

ModelStage=Callable[[Manager,
                     WikiTFR,
                     Optional[int],
                     Optional[int],
                     Optional[RRef]],BertPretrain]

def model_3()->ModelStage:
  def _nc(c):
    mklens(c).name.val+='-3L'
    mklens(c).bert_config_template.num_hidden_layers.val=3
    mklens(c).train_batch_size.val=128
  return redefine(minibert_pretrain_wiki, new_config=_nc)

def model_6()->ModelStage:
  def _nc(c):
    mklens(c).name.val+='-6L'
    mklens(c).bert_config_template.num_hidden_layers.val=6
    mklens(c).train_batch_size.val=64
  return redefine(minibert_pretrain_wiki, new_config=_nc)

def calculate_pretrain_epoches(stage_ds:DatasetStage,
                               train_batch_size:int,
                               train_steps_per_epoch:int=DEF_STEPS_PER_EPOCH)->int:
  """ Calculate the number of epoches required to match the reference model in
  the number of parameter updates.

  Ref. https://arxiv.org/pdf/1810.04805.pdf, section A.2. "Pre-training
  procedure"
  """
  upstream_train_steps=10**6
  upstream_batch_size=256
  upstream_seq_length=512
  upstream_total_tokens=upstream_train_steps * upstream_batch_size * upstream_seq_length

  # Calculate number of training epoches for our model to match the upstream
  our_batch_size=train_batch_size
  our_seq_length=mklens(instantiate(stage_ds).dref).max_seq_length.val
  our_train_steps=upstream_total_tokens / (our_batch_size * our_seq_length)
  out_epoches=our_train_steps // train_steps_per_epoch
  return out_epoches


def experiment_pretrain(model:ModelStage,
                        ds:DatasetStage,
                        nepoches:int=DEF_NEPOCHES,
                        train_steps_per_epoch:int=DEF_STEPS_PER_EPOCH,
                        epoches_step:int=DEF_EPOCHES_BETWEEN_FINETUNES,
                        finetune_task_name:str=DEF_FINETUNE_TASK,
                        )->Tuple[dict,dict]:
  """ Pretrain BERT for `train_steps_per_epoch*nepoches` steps on `ds` corups.
  Pause every `epoches_step` epoches to make a fine-tuning on the GLUE task of
  `finetune_task_name`. Return a tuple of dicts, mapping number of pre-trained
  epoches to fine-tuning realization references.
  """
  assert finetune_task_name in glue_tasks()

  def _pretrain_stage(nepoch:int, resume_rref:Optional[RRef])->Stage:
    def _stage(m):
      return model(m,
        tfrecs=ds(m),
        train_steps_per_epoch=train_steps_per_epoch,
        train_epoches=nepoch,
        resume_rref=resume_rref)
    return _stage

  def _finetune_stage(nepoch:int)->Stage:
    def _stage(m)->BertGlue:
      refglue=all_fetchglue(m)
      refbert=_pretrain_stage(nepoch, None)(m)
      gluetfr=glue_tfrecords(m,
        finetune_task_name,
        bert_vocab=mklens(refbert).bert_vocab.refpath,
        lower_case=(mklens(refbert).cased.val==False),
        refdataset=refglue)
      tfbert=bert_finetune_glue(m,refbert,gluetfr)
      return tfbert
    return _stage

  pretrained:Dict[int,RRef]={}
  finetuned:Dict[int,RRef]={}
  for e in range(epoches_step,nepoches+epoches_step,epoches_step):
    print(f"Pre-training up to {e}/{nepoches}")
    pretrained[e]=realize(instantiate(
      _pretrain_stage(e, pretrained.get(e-epoches_step))))
    linkrref(pretrained[e],['bert_pretrain',f'epoch-{e}'], verbose=True)
    print(f"Fine-tunining after {e}-epoch pre-training")
    finetuned[e]=realize(instantiate(_finetune_stage(e)))
    linkrref(finetuned[e],['bert_pretrain',f'epoch-{e}'], verbose=True)
  return pretrained,finetuned




if __name__== '__main__':
  print(experiment_dataset_en())
  print(experiment_dataset_ru())
  print(experiment_pretrain(model_6(), dataset_en))
  print(experiment_pretrain(model_3(), dataset_en))


