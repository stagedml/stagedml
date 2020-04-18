import os
import tensorflow as tf
assert tf.version.VERSION.startswith('2.1') or \
       tf.version.VERSION.startswith('2.2')

import json
from official.nlp.bert.run_classifier import get_loss_fn
from official.nlp.bert.configs import BertConfig
from official.modeling.model_training_utils import run_customized_training_loop
from official.nlp.optimization import create_optimizer
from tensorflow.python.keras.backend import clear_session
from tensorflow.keras.callbacks import TensorBoard
from absl import logging

from pylightnix import ( Build, Path, Config, Manager, RRef, DRef, Context,
    store_cattrs, build_outpath, build_cattrs, mkdrv, rref2path, json_load,
    build_config, mklens, build_wrapper_, mkconfig, promise )

from stagedml.datasets.glue.tfdataset import ( dataset, dataset_eval, dataset_train )
from stagedml.models.bert import ( BertLayer, BertInput, BertOutput,
    BertModel, classification_logits )
from stagedml.utils.tf import ( runtb, runtensorboard, thash, dpurge, modelhash )
from stagedml.core import ( protocol_add, protocol_add_hist,
    protocol_add_eval, protocol_match )
from stagedml.types import ( BertCP, GlueTFR, BertGlue, Optional,Any,List,Tuple,Union )


class ModelBuild(Build):
  model:tf.keras.Model
  model_eval:tf.keras.Model
  core_model:tf.keras.Model
  strategy:Any
  optimizer:Any


def build(b:ModelBuild, clear_session:bool=True):
  tf.keras.backend.clear_session()

  c = build_cattrs(b)

  with open(mklens(b).bert_config.syspath, "r") as f:
    bert_config = BertConfig.from_dict(json_load(f))

  with open(mklens(b).task_config.syspath, "r") as f:
    task_config = json_load(f)

  c.num_labels = int(task_config['num_classes'])
  c.max_seq_length = int(task_config['max_seq_length'])

  c.train_batch_size = c.batch_size
  c.eval_batch_size = c.batch_size
  if c.dataset_size is None:
    c.dataset_size = int(task_config['train_data_size'])
  c.train_data_size = int(c.dataset_size*0.95)
  c.valid_data_size = int(c.dataset_size)-c.train_data_size
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


    teacher_ins = BertInput(input_word_ids, input_mask, input_type_ids)
    teacher = BertLayer(config=bert_config, float_type=tf.float32, name='bert')
    teacher_model = BertModel(teacher_ins, teacher(teacher_ins))
    teacher_model.model.summary()

    teacher_outs = teacher_model(teacher_ins)
    teacher_cls_logits = classification_logits(config=bert_config,
                                               num_labels=c.num_labels,
                                               pooled_input=teacher_outs.cls_output)

    teacher_cls_probs = tf.keras.layers.Activation('softmax')(teacher_cls_logits)

    model = tf.keras.Model(inputs=teacher_ins, outputs=[teacher_cls_logits])
    model_eval = tf.keras.Model(inputs=teacher_ins, outputs=[teacher_cls_probs])

    b.optimizer = create_optimizer(c.lr, c.train_steps_per_epoch*c.train_epoches, c.train_warmup_steps)

    b.model = model
    b.core_model = teacher_model.model
    b.model_eval = model_eval
  return


def cpload(b:ModelBuild)->None:
  """ Load checkpoint into model """
  c = build_cattrs(b)
  checkpoint = tf.train.Checkpoint(model=b.core_model)
  checkpoint.restore(mklens(b).bert_ckpt.syspath).assert_consumed()
  protocol_add(mklens(b).protocol.syspath, 'cpload')


def train(b:ModelBuild, **kwargs)->None:
  """ Train the model by using `fit` method of Keras.model """
  assert b.model is not None
  c = build_cattrs(b)
  o = build_outpath(b)
  with b.strategy.scope():

    dt, dv = dataset_train(mklens(b).task_train.syspath, c.train_data_size,
                           c.train_batch_size, c.max_seq_length)
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
    b.model.save_weights(mklens(b).weights.syspath, save_format='h5')
    protocol_add_hist(mklens(b).protocol.syspath, 'train', modelhash(b.model), h)

def evaluate(b:ModelBuild)->None:
  c = build_cattrs(b)
  o = build_outpath(b)
  print('Evaluating')

  with b.strategy.scope():
    metric_fn = tf.keras.metrics.SparseCategoricalAccuracy('eval_accuracy', dtype=tf.float32)
    loss_fn = get_loss_fn(num_classes=int(c.num_labels), loss_factor=1.0)
    k = b.model_eval
    k.compile(b.optimizer, loss=loss_fn, metrics=[metric_fn])

    de = dataset_eval(mklens(b).task_eval.syspath, c.eval_batch_size, c.max_seq_length)
    h = k.evaluate(de, steps=c.eval_steps_per_epoch)

    filewriter = tf.summary.create_file_writer(o)
    with filewriter.as_default():
      for mname,v in zip(k.metrics_names,h):
        tf.summary.scalar(mname,v,step=0)

    protocol_add_eval(mklens(b).protocol.syspath, 'evaluate', modelhash(b.model), k.metrics_names, h)

def bert_finetune_realize(b:ModelBuild)->None:
  build(b); cpload(b); train(b); evaluate(b)

def bert_finetune_glue(m:Manager, refbert:BertCP, tfrecs:GlueTFR)->BertGlue:
  def _config()->dict:
    nonlocal tfrecs
    name = 'bert-finetune-'+mklens(tfrecs).task_name.val.lower()

    task_train = mklens(tfrecs).outputs.train.refpath
    task_eval = mklens(tfrecs).outputs.dev.refpath
    task_config = mklens(tfrecs).outputs.meta.refpath
    bert_config = mklens(refbert).bert_config.refpath
    bert_ckpt = mklens(refbert).bert_ckpt.refpath
    assert mklens(refbert).bert_vocab.refpath==mklens(tfrecs).bert_vocab.refpath

    protocol = [promise, 'protocol.json']
    weights = [promise, 'weights.h5']
    dataset_size = None # Use default value

    lr = 2e-5
    batch_size = 8
    train_epoches = 3
    version = 4
    return locals()

  return BertGlue(mkdrv(m,
    config=mkconfig(_config()),
    matcher=protocol_match('evaluate', 'eval_accuracy'),
    realizer=build_wrapper_(bert_finetune_realize, ModelBuild)))



