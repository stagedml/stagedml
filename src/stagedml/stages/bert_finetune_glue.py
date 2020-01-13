import os
import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

import json
from official.nlp.bert.run_classifier import get_loss_fn
from official.nlp.bert_modeling import BertConfig
from official.modeling.model_training_utils import run_customized_training_loop
from official.nlp.optimization import create_optimizer
from tensorflow.python.keras.backend import clear_session
from tensorflow.keras.callbacks import TensorBoard
from absl import logging

from typing import Optional,Any,List,Tuple,Union
from pylightnix import ( Config, State, protocol_add, search, model_config,
                         config_ro, model_config_ro, Ref, store_refpath,
                         config_deref, state, state_add, store_readjson,
                         store_systempath, model_outpath )

from stagedml.utils.instantiate import Options, instantiate
from stagedml.dataset.glue.tfdataset import dataset, dataset_eval, dataset_train
from stagedml.model.tiny_bert import ( TeacherBertModel as TeacherBert,
                                       classification_logits )
from stagedml.utils.tf import ( runtb, runtensorboard, thash, save, KerasModel,
                                protocol_add_hist, protocol_add_eval, best,
                                dpurge, best )
from stagedml.utils.refs import GlueTFR, BertGlue

refpath = store_refpath

class Model(KerasModel):
  model:tf.keras.Model
  model_eval:tf.keras.Model
  core_model:tf.keras.Model
  strategy:Any
  optimizer:Any

def config(task_name:str, tfrecs:GlueTFR)->Config:
  name = 'bert-finetune'

  task_train_refpath = refpath(tfrecs, [task_name,'train.tfrecord'])
  task_eval_refpath = refpath(tfrecs, [task_name, 'dev.tfrecord'])
  task_config = store_readjson(refpath(tfrecs, [task_name, 'meta.json']))
  num_labels = int(task_config['num_classes'])

  bert_ckpt_refpath = config_ro(config_deref(tfrecs))['bert_ckpt_refpath']
  bert_config = config_ro(config_deref(tfrecs))['bert_config']

  lr = 2e-5
  batch_size = 8
  train_epoches = 3
  train_batch_size = batch_size
  eval_batch_size = batch_size
  train_data_size = int(task_config['train_data_size']*0.95)
  valid_data_size = int(task_config['train_data_size'])-train_data_size
  eval_data_size = task_config['dev_data_size']
  train_steps_per_epoch = int(train_data_size / train_batch_size)
  valid_steps_per_epoch = int(valid_data_size / train_batch_size)
  eval_steps_per_epoch = int(eval_data_size / eval_batch_size)
  train_warmup_steps = int(train_epoches * train_data_size * 0.1 / train_batch_size)
  version=2
  return Config({k:v for k,v in locals().items() if k[0]!='_'})

def build(m:Model, clear_session:bool=True):
  tf.keras.backend.clear_session()
  c = model_config_ro(m)
  bert_config = BertConfig.from_dict(c.bert_config)
  num_labels = c.num_labels
  max_seq_length = c.task_config['max_seq_length']

  m.strategy=tf.distribute.MirroredStrategy()
  with m.strategy.scope():
    input_word_ids = tf.keras.Input(shape=(max_seq_length,), name='input_word_ids', dtype=tf.int32)
    input_mask     = tf.keras.Input(shape=(max_seq_length,), name='input_mask', dtype=tf.int32)
    input_type_ids = tf.keras.Input(shape=(max_seq_length,), name='input_type_ids', dtype=tf.int32)
    inputs = {
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids
        }

    teacher = TeacherBert(config=bert_config, float_type=tf.float32, name='teacher')
    teacher_outs = teacher(input_word_ids, input_mask, input_type_ids)
    teacher_model = tf.keras.Model(inputs=inputs, outputs=teacher_outs)
    teacher_model.summary()

    pooled_output,_,_,_ = teacher_model(inputs)
    teacher_cls_logits = classification_logits(config=bert_config,
                                               num_labels=c.num_labels,
                                               pooled_input=pooled_output)

    teacher_cls_probs = tf.keras.layers.Activation('softmax')(teacher_cls_logits)

    model = tf.keras.Model(inputs=inputs, outputs=[teacher_cls_logits])
    model_eval = tf.keras.Model(inputs=inputs, outputs=[teacher_cls_probs])

    m.optimizer = create_optimizer(c.lr, c.train_steps_per_epoch*c.train_epoches, c.train_warmup_steps)

    m.model = model
    m.core_model = teacher_model
    m.model_eval = model_eval
  return m


def cploaded(s:State)->State:
  return state_add(s,'cpload')
def cpload(m:Model)->Model:
  """ Load checkpoint into model """
  c = model_config_ro(m)
  checkpoint = tf.train.Checkpoint(model=m.core_model)
  checkpoint.restore(store_systempath(c.bert_ckpt_refpath)).assert_consumed()
  protocol_add(m, 'cpload')
  return m


def trained(s:State)->State:
  return state_add(s, 'train')
def train(m:Model, **kwargs)->Model:
  """ Train the model by using `fit` method of Keras.model """
  assert m.model is not None
  c = model_config_ro(m)

  dt, dv = dataset_train(store_systempath(c.task_train_refpath),  model_config(m))
  o = model_outpath(m)
  tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
                                     write_graph=False, update_freq='batch')

  print('Training')
  loss_fn = get_loss_fn(num_classes=c.num_labels, loss_factor=1.0)
  metric_fn = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)
  m.model.compile(m.optimizer, loss=loss_fn, metrics=[metric_fn])
  h = m.model.fit(
    dt,
    steps_per_epoch=c.train_steps_per_epoch,
    validation_data=dv,
    validation_steps=c.valid_steps_per_epoch,
    epochs=c.train_epoches,
    callbacks=[tensorboard_callback])
  protocol_add_hist(m, 'train', h)
  return m


def ctrained(s:State)->State:
  return state_add(s, 'ctrain')
def ctrain(m:Model)->Model:
  """ Train the model by using Custom training loop from TF Official models """
  assert m.model is not None
  c = model_config_ro(m)
  print('Training (custom)')

  loss_fn = get_loss_fn(num_classes=c.num_labels, loss_factor=1.0)

  def _metric_fn():
    return \
      tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy', dtype=tf.float32)

  def _get_model():
    return m.model,None

  def _train_input_fn(is_training:bool=True, drop_remainder:bool=True)->Any:
    d = dataset(store_systempath(c.task_train_refpath), model_config(m))
    dtrain=d.take(c.train_data_size)
    if is_training:
      dtrain=dtrain.shuffle(100)
      dtrain=dtrain.repeat()
    dtrain=dtrain.batch(c.train_batch_size, drop_remainder=drop_remainder)
    dtrain=dtrain.prefetch(1024)
    return dtrain

  def _eval_input_fn()->Any:
    d = dataset(store_systempath(c.task_train_refpath), model_config(m))
    dvalid=d.skip(c.train_data_size)
    dvalid=dvalid.batch(c.train_batch_size, drop_remainder=False)
    return dvalid

  o = model_outpath(m)
  tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
                                     write_graph=False, update_freq='batch')
  m.model.optimizer = m.optimizer
  logging.set_verbosity(logging.INFO)
  run_customized_training_loop(
      strategy=m.strategy,
      model_fn=_get_model,
      loss_fn=loss_fn,
      model_dir=o,
      steps_per_epoch=c.train_steps_per_epoch,
      steps_per_loop=2*c.batch_size,
      epochs=c.train_epoches,
      train_input_fn=_train_input_fn,
      eval_input_fn=_eval_input_fn,
      eval_steps=c.valid_steps_per_epoch,
      # init_checkpoint=ckpt,
      metric_fn=_metric_fn,
      custom_callbacks=None, # [tensorboard_callback],
      run_eagerly=False)

  dpurge(o,'ctl_step.*ckpt', debug=True)
  with open(o+'/summaries/training_summary.txt', 'r') as f:
    s=json.load(f)
  protocol_add(m, 'ctrain', result=s)
  return m



def evaluated(s:State)->State:
  return state_add(s, 'evaluate')
def evaluate(m:Model)->Model:
  c = model_config_ro(m)
  print('Evaluating')

  metric_fn = tf.keras.metrics.SparseCategoricalAccuracy('eval_accuracy', dtype=tf.float32)
  loss_fn = get_loss_fn(num_classes=int(c.num_labels), loss_factor=1.0)
  k = m.model_eval
  k.compile(m.optimizer, loss=loss_fn, metrics=[metric_fn])

  de = dataset_eval(store_systempath(c.task_eval_refpath), model_config(m))
  h = k.evaluate(de, steps=c.eval_steps_per_epoch)

  o=model_outpath(m)
  filewriter = tf.summary.create_file_writer(o)
  with filewriter.as_default():
    for mname,v in zip(k.metrics_names,h):
      tf.summary.scalar(mname,v,step=0)
  protocol_add_eval(m, 'evaluate', k.metrics_names, h)
  return m


def bert_finetune(o:Options, task_name:str, tfrecs:GlueTFR)->BertGlue:
  c=config(task_name, tfrecs)
  def _search():
    return search(evaluated(ctrained(cploaded(state(c)))))
  def _build():
    m=Model(c)
    build(m)
    cpload(m)
    ctrain(m)
    evaluate(m)
    return save(m)
  return BertGlue(instantiate(o, _search, _build))


