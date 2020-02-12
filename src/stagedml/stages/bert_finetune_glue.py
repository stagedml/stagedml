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
from pylightnix import ( Config, Manager, RRef, DRef, store_cattrs, build_path,
    build_outpath, build_cattrs, mkdrv, rref2path, json_load, build_config,
    mkbuild )

from stagedml.datasets.glue.tfdataset import ( dataset, dataset_eval, dataset_train )
from stagedml.models.bert import ( BertLayer, classification_logits )
from stagedml.utils.tf import ( runtb, runtensorboard, thash, KerasBuild,
    protocol_add, protocol_add_hist, protocol_add_eval, match_metric, dpurge,
    keras_save )
from stagedml.utils.refs import ( GlueTFR, BertGlue )


def config(task_name:str, tfrecs:GlueTFR)->Config:
  name = 'bert-finetune'

  task_train_refpath = [tfrecs, task_name,'train.tfrecord']
  task_eval_refpath = [tfrecs, task_name, 'dev.tfrecord']
  task_config_refpath = [tfrecs, task_name, 'meta.json']

  bert_config_refpath = store_cattrs(tfrecs).bert_config_refpath
  bert_ckpt_refpath = store_cattrs(tfrecs).bert_ckpt_refpath

  lr = 2e-5
  batch_size = 8
  train_epoches = 3
  version = 3
  return Config(locals())


class ModelBuild(KerasBuild):
  model:tf.keras.Model
  model_eval:tf.keras.Model
  core_model:tf.keras.Model
  strategy:Any
  optimizer:Any


def build(b:ModelBuild, clear_session:bool=True):
  tf.keras.backend.clear_session()

  c = build_cattrs(b)

  with open(build_path(b, c.bert_config_refpath), "r") as f:
    bert_config = BertConfig.from_dict(json_load(f))

  with open(build_path(b, c.task_config_refpath), "r") as f:
    task_config = json_load(f)

  c.num_labels = int(task_config['num_classes'])
  c.max_seq_length = int(task_config['max_seq_length'])

  c.train_batch_size = c.batch_size
  c.eval_batch_size = c.batch_size
  c.train_data_size = int(task_config['train_data_size']*0.95)
  c.valid_data_size = int(task_config['train_data_size'])-c.train_data_size
  c.eval_data_size = task_config['dev_data_size']
  c.train_steps_per_epoch = int(c.train_data_size / c.train_batch_size)
  c.valid_steps_per_epoch = int(c.valid_data_size / c.train_batch_size)
  c.eval_steps_per_epoch = int(c.eval_data_size / c.eval_batch_size)
  c.train_warmup_steps = int(c.train_epoches * c.train_data_size * 0.1 / c.train_batch_size)

  b.strategy=tf.distribute.MirroredStrategy()
  with b.strategy.scope():
    input_word_ids = tf.keras.Input(shape=(c.max_seq_length,), name='input_word_ids', dtype=tf.int32)
    input_mask     = tf.keras.Input(shape=(c.max_seq_length,), name='input_mask', dtype=tf.int32)
    input_type_ids = tf.keras.Input(shape=(c.max_seq_length,), name='input_type_ids', dtype=tf.int32)
    inputs = {
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids
        }

    teacher = BertLayer(config=bert_config, float_type=tf.float32, name='bert')
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

    b.optimizer = create_optimizer(c.lr, c.train_steps_per_epoch*c.train_epoches, c.train_warmup_steps)

    b.model = model
    b.core_model = teacher_model
    b.model_eval = model_eval
  return


def cpload(b:ModelBuild)->None:
  """ Load checkpoint into model """
  c = build_cattrs(b)
  checkpoint = tf.train.Checkpoint(model=b.core_model)
  checkpoint.restore(build_path(b, c.bert_ckpt_refpath)).assert_consumed()
  protocol_add(b, 'cpload')
  return


def train(b:ModelBuild, **kwargs)->None:
  """ Train the model by using `fit` method of Keras.model """
  assert b.model is not None
  c = build_cattrs(b)
  o = build_outpath(b)

  dt, dv = dataset_train(build_path(b, c.task_train_refpath), c.train_data_size, c.train_batch_size, c.max_seq_length)
  tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
                                     write_graph=False, update_freq='batch')

  print('Training')
  loss_fn = get_loss_fn(num_classes=c.num_labels, loss_factor=1.0)
  metric_fn = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)
  b.model.compile(b.optimizer, loss=loss_fn, metrics=[metric_fn])
  h = b.model.fit(
    dt,
    steps_per_epoch=c.train_steps_per_epoch,
    validation_data=dv,
    validation_steps=c.valid_steps_per_epoch,
    epochs=c.train_epoches,
    callbacks=[tensorboard_callback])
  protocol_add_hist(b, 'train', h)
  return


# def ctrain(m:Model)->Model:
#   """ Train the model by using Custom training loop from TF Official models """
#   assert m.model is not None
#   c = model_config_ro(m)
#   print('Training (custom)')

#   loss_fn = get_loss_fn(num_classes=c.num_labels, loss_factor=1.0)

#   def _metric_fn():
#     return \
#       tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy', dtype=tf.float32)

#   def _get_model():
#     return m.model,None

#   def _train_input_fn(is_training:bool=True, drop_remainder:bool=True)->Any:
#     d = dataset(store_systempath(c.task_train_refpath), model_config(m))
#     dtrain=d.take(c.train_data_size)
#     if is_training:
#       dtrain=dtrain.shuffle(100)
#       dtrain=dtrain.repeat()
#     dtrain=dtrain.batch(c.train_batch_size, drop_remainder=drop_remainder)
#     dtrain=dtrain.prefetch(1024)
#     return dtrain

#   def _eval_input_fn()->Any:
#     d = dataset(store_systempath(c.task_train_refpath), model_config(m))
#     dvalid=d.skip(c.train_data_size)
#     dvalid=dvalid.batch(c.train_batch_size, drop_remainder=False)
#     return dvalid

#   o = model_outpath(m)
#   tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
#                                      write_graph=False, update_freq='batch')
#   m.model.optimizer = m.optimizer
#   logging.set_verbosity(logging.INFO)
#   run_customized_training_loop(
#       strategy=m.strategy,
#       model_fn=_get_model,
#       loss_fn=loss_fn,
#       model_dir=o,
#       steps_per_epoch=c.train_steps_per_epoch,
#       steps_per_loop=2*c.batch_size,
#       epochs=c.train_epoches,
#       train_input_fn=_train_input_fn,
#       eval_input_fn=_eval_input_fn,
#       eval_steps=c.valid_steps_per_epoch,
#       # init_checkpoint=ckpt,
#       metric_fn=_metric_fn,
#       custom_callbacks=None, # [tensorboard_callback],
#       run_eagerly=False)

#   dpurge(o,'ctl_step.*ckpt', debug=True)
#   with open(o+'/summaries/training_summary.txt', 'r') as f:
#     s=json.load(f)
#   protocol_add(m, 'ctrain', result=s)
#   return m


def evaluate(b:ModelBuild):
  c = build_cattrs(b)
  o = build_outpath(b)
  print('Evaluating')

  metric_fn = tf.keras.metrics.SparseCategoricalAccuracy('eval_accuracy', dtype=tf.float32)
  loss_fn = get_loss_fn(num_classes=int(c.num_labels), loss_factor=1.0)
  k = b.model_eval
  k.compile(b.optimizer, loss=loss_fn, metrics=[metric_fn])

  de = dataset_eval(build_path(b, c.task_eval_refpath), c.eval_batch_size, c.max_seq_length)
  h = k.evaluate(de, steps=c.eval_steps_per_epoch)

  filewriter = tf.summary.create_file_writer(o)
  with filewriter.as_default():
    for mname,v in zip(k.metrics_names,h):
      tf.summary.scalar(mname,v,step=0)
  protocol_add_eval(b, 'evaluate', k.metrics_names, h)
  return


def bert_finetune_glue(m:Manager, task_name:str, tfrecs:GlueTFR)->BertGlue:
  def _realize(dref,context):
    b=ModelBuild(mkbuild(dref,context));
    build(b); cpload(b); train(b); evaluate(b); keras_save(b)
    return build_outpath(b)
  return BertGlue(mkdrv(m,
    config=config(task_name, tfrecs),
    matcher=match_metric('evaluate', 'eval_accuracy'),
    realizer=_realize))




