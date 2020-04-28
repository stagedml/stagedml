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
    store_cattrs, build_outpaths, build_cattrs, mkdrv, rref2path, json_load,
    build_config, mklens, build_wrapper_, mkconfig, promise, claim,
    build_setoutpaths )

from stagedml.datasets.glue.tfdataset import ( dataset, dataset_eval,
    dataset_train, dataset_valid )
from stagedml.models.bert import ( BertLayer, BertInput, BertOutput,
    BertModel, classification_logits )
from stagedml.imports.tf import ( load_checkpoint, NotFoundError, Tensor, Mean,
    SparseCategoricalAccuracy )
from stagedml.imports.sys import ( join )
from stagedml.utils.tf import ( runtb, runtensorboard, thash, dpurge,
    modelhash, print_model_checkpoint_diff )
from stagedml.core import ( protocol_add, protocol_add_hist,
    protocol_add_eval, protocol_match )
from stagedml.types import ( BertCP, GlueTFR, BertGlue,
    Optional,Any,List,Tuple,Union )


class ModelBuild(Build):
  model:tf.keras.Model
  model_eval:tf.keras.Model
  core_model:tf.keras.Model
  strategy:Any
  optimizer:Any


def build(b:ModelBuild, iid:int=0, clear_session:bool=True):
  tf.keras.backend.clear_session()

  c=build_cattrs(b)
  l=mklens(b,build_output_idx=iid)

  with open(l.bert_config.syspath, "r") as f:
    bert_config = BertConfig.from_dict(json_load(f))

  with open(l.task_config.syspath, "r") as f:
    task_config = json_load(f)

  c.num_labels = int(task_config['num_classes'])
  c.max_seq_length = int(task_config['max_seq_length'])

  c.train_batch_size = c.train_batch_size
  c.eval_batch_size = c.eval_batch_size
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



def cpload(b:ModelBuild, iid:int=0)->None:
  """ Load checkpoint into model
  FIXME: invent a more productive approach to loading checkpoints """
  c=build_cattrs(b)
  o=build_outpaths(b)[iid]
  l=mklens(b,build_output_idx=iid)
  exceptions=[]

  try:
    checkpoint=tf.train.Checkpoint(model=b.core_model)
    load_status=checkpoint.restore(l.bert_ckpt_in.syspath)
    load_status.assert_consumed()
    protocol_add(l.protocol.syspath, 'cpload:1', modelhash(b.model))
    return
  except Exception as e:
    exceptions.append(e)

  try:
    r=load_checkpoint(l.bert_ckpt_in.syspath)
    for w in b.core_model.weights:
      w.assign(r.get_tensor(w.name.split(':')[0]))
    protocol_add(l.protocol.syspath, 'cpload:2', modelhash(b.model))
    return
  except Exception as e:
    exceptions.append(e)

  try:
    r=load_checkpoint(l.bert_ckpt_in.syspath)
    for w in b.core_model.weights:
      name=w.name.split(':')[0]
      w.assign(r.get_tensor({
        'bert/word_embeddings/embeddings':'bert/embeddings/word_embeddings'}\
            .get(name,name)))
    protocol_add(l.protocol.syspath, 'cpload:3', modelhash(b.model))
    return
  except Exception as e:
    exceptions.append(e)

  print_model_checkpoint_diff(b.core_model, l.bert_ckpt_in.syspath, o)
  print("Unused variables:")
  print('\n'.join(['- '+w.name.split(':')[0]
                   for w in load_status._checkpoint.unused_attributes.keys()]))
  assert False, f"None of the checkpoint loading methods have succeed"



def train(b:ModelBuild, iid:int=0)->None:
  """ Train the model by using `fit` method of Keras.model """
  assert b.model is not None
  c=build_cattrs(b)
  o=build_outpaths(b)[iid]
  l=mklens(b,build_output_idx=iid)
  with b.strategy.scope():

    dt = dataset_train(l.task_train.syspath,
                       c.train_data_size,
                       train_batch_size=c.train_batch_size,
                       max_seq_length=c.max_seq_length)
    dv = dataset_valid(l.task_train.syspath,
                       c.train_data_size,
                       valid_batch_size=c.train_batch_size,
                       max_seq_length=c.max_seq_length)

    tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
                                       write_graph=False, update_freq='batch')

    print('Training')
    loss_fn = get_loss_fn(num_classes=c.num_labels, loss_factor=1.0)
    metric_fn = SparseCategoricalAccuracy('accuracy', dtype=tf.float32)
    b.model.compile(b.optimizer, loss=loss_fn, metrics=[metric_fn])
    h = b.model.fit(
      dt,
      steps_per_epoch=c.train_steps_per_epoch,
      validation_data=dv,
      validation_steps=c.valid_steps_per_epoch,
      epochs=c.train_epoches,
      callbacks=[tensorboard_callback])
    b.model.save_weights(l.weights.syspath, save_format='h5')
    protocol_add_hist(l.protocol.syspath, 'train', modelhash(b.model), h)

def train_custom(b:ModelBuild, iid:int=0):
  c=build_cattrs(b)
  o=build_outpaths(b)[iid]
  l=mklens(b,build_output_idx=iid)

  train_summary_writer = tf.summary.create_file_writer(join(o,'train'))
  valid_summary_writer = tf.summary.create_file_writer(join(o,'valid'))
  loss_metric = Mean('loss', dtype=tf.float32)
  metrics = [ SparseCategoricalAccuracy('accuracy', dtype=tf.float32) ]
  loss_fn=get_loss_fn(num_classes=c.num_labels, loss_factor=1.0)
  b.model.compile(b.optimizer, loss=loss_fn, metrics=metrics)

  def _train_input_fn(ctx:Any)->Any:
    global_batch_size=c.train_batch_size
    batch_size=ctx.get_per_replica_batch_size(global_batch_size) \
               if ctx else global_batch_size
    return dataset_train(l.task_train.syspath, c.train_data_size,
                         batch_size, c.max_seq_length)

  def _valid_input_fn(ctx:Any)->Any:
    global_batch_size=c.train_batch_size
    batch_size=ctx.get_per_replica_batch_size(global_batch_size) \
               if ctx else global_batch_size
    return dataset_valid(l.task_train.syspath, c.train_data_size,
                         batch_size, c.max_seq_length)

  def _metrics_reset()->None:
    for m in b.model.metrics + [loss_metric]:
      m.reset_states()

  def _metrics_update(labels:Tensor, inputs:Tensor,
                      loss:Optional[Tensor]=None)->None:
    for m in b.model.metrics:
      m.update_state(labels, inputs)
    if loss is not None:
      loss_metric.update_state(loss)

  def _metrics_dump(writer, step:int)->None:
    from tensorflow.python.ops.math_ops import cast
    from tensorflow.python.framework import dtypes
    with writer.as_default():
      for m in b.model.metrics + [loss_metric]:
        tf.summary.scalar(m.name, cast(m.result(), dtypes.float32), step=step)
      writer.flush()


  with b.strategy.scope():
    def _train_step_replicated(inputs):
      inputs, labels = inputs
      with tf.GradientTape() as tape:
        outs=b.model(inputs, training=True)
        loss=loss_fn(labels, outs)
      tv=b.model.trainable_variables
      grad=tape.gradient(loss,tv)
      b.model.optimizer.apply_gradients(zip(grad,tv))
      _metrics_update(labels, outs, loss)

    @tf.function
    def _train_step(iterator):
      b.strategy.experimental_run_v2(
        _train_step_replicated, args=(next(iterator),))

    def _valid_step_replicated(inputs):
      inputs, labels = inputs
      outs=b.model(inputs, training=False)
      loss=loss_fn(labels, outs)
      _metrics_update(labels, outs, loss)

    @tf.function
    def _valid_step(iterator):
      b.strategy.experimental_run_v2(
        _valid_step_replicated, args=(next(iterator),))


    train_iterator=iter(
      b.strategy.experimental_distribute_datasets_from_function(
        _train_input_fn))
    current_step=b.model.optimizer.iterations.numpy()
    while current_step<c.train_epoches*c.train_steps_per_epoch:

      next_epoch=(current_step//c.train_steps_per_epoch)+1
      print(f"Next epoch is {next_epoch}")
      while current_step<next_epoch*c.train_steps_per_epoch:
        print(f"Current step is {current_step}/{next_epoch*c.train_steps_per_epoch}")
        _metrics_reset()
        _train_step(train_iterator)
        current_step += 1
        _metrics_dump(train_summary_writer, current_step)

      valid_iterator=iter(
        b.strategy.experimental_distribute_datasets_from_function(
          _valid_input_fn))
      _metrics_reset()
      for step in range(c.valid_steps_per_epoch):
        print(f"Valid step is {step}/{c.valid_steps_per_epoch}")
        _valid_step(valid_iterator)
      _metrics_dump(valid_summary_writer, current_step)

      print(f"Saving '{l.bert_ckpt.syspath}' after {current_step} step")
      b.core_model.save_weights(l.bert_ckpt.syspath)
      print(f"Saving '{l.checkpoint_full.syspath}' after {current_step} step")
      b.model.save_weights(l.checkpoint_full.syspath)

  protocol_add(l.protocol.syspath, 'train', modelhash(b.model))


def evaluate(b:ModelBuild, iid:int=0)->None:
  c=build_cattrs(b)
  o=build_outpaths(b)[iid]
  l=mklens(b,build_output_idx=iid)
  print('Evaluating')

  with b.strategy.scope():
    metric_fn = SparseCategoricalAccuracy('eval_accuracy', dtype=tf.float32)
    loss_fn = get_loss_fn(num_classes=int(c.num_labels), loss_factor=1.0)
    k = b.model_eval
    k.compile(b.optimizer, loss=loss_fn, metrics=[metric_fn])

    de = dataset_eval(l.task_eval.syspath, c.eval_batch_size, c.max_seq_length)
    h = k.evaluate(de, steps=c.eval_steps_per_epoch)

    filewriter = tf.summary.create_file_writer(join(o,'eval'))
    with filewriter.as_default():
      for mname,v in zip(k.metrics_names,h):
        tf.summary.scalar(mname,v.astype(float),step=0)

    protocol_add_eval(l.protocol.syspath, 'evaluate',
                      modelhash(b.model), k.metrics_names, h)

def bert_finetune_glue(m:Manager, refbert:BertCP,
                       tfrecs:GlueTFR, num_instances:int=1)->BertGlue:

  def _realize(b:ModelBuild)->None:
    build_setoutpaths(b,num_instances)
    for i in range(num_instances):
      print(f"Training instance {i}")
      build(b,i); cpload(b,i); train_custom(b,i); evaluate(b,i)

  def _config()->dict:
    nonlocal tfrecs
    name = 'bert-finetune-'+mklens(tfrecs).task_name.val.lower()

    task_train = mklens(tfrecs).outputs.train.refpath
    task_eval = mklens(tfrecs).outputs.dev.refpath
    task_config = mklens(tfrecs).outputs.meta.refpath
    bert_config = mklens(refbert).bert_config.refpath
    bert_ckpt_in = mklens(refbert).bert_ckpt.refpath
    assert mklens(refbert).bert_vocab.refpath==mklens(tfrecs).bert_vocab.refpath, \
        "Model dictionary path doesn't match the dataset dictionary path"

    checkpoint_full = [claim, 'checkpoint_full.ckpt']
    bert_ckpt = [claim, 'checkpoint_bert.ckpt']
    protocol = [promise, 'protocol.json']
    # weights = [promise, 'weights.h5']
    dataset_size = None # Use default value

    lr = 2e-5
    train_batch_size = 32
    eval_batch_size = 32
    train_epoches = 3
    version = 8
    return locals()

  return BertGlue(mkdrv(m,
    config=mkconfig(_config()),
    matcher=protocol_match('evaluate', 'eval_accuracy'),
    realizer=build_wrapper_(_realize, ModelBuild)))



