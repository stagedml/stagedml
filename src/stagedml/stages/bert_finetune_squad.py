import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from os.path import join
from absl import logging
from tensorflow.python.keras.backend import clear_session
from tensorflow.keras.callbacks import TensorBoard

from official.nlp.bert_modeling import BertConfig
from official.nlp.optimization import create_optimizer
from official.modeling.model_training_utils import run_customized_training_loop

from pylightnix import ( Path, Manager, Config, DRef, RRef, Context, match_only,
    store_cattrs, build_cattrs, build_path, build_outpath, json_load, mkbuild,
    mkdrv, match_latest )

from stagedml.datasets.glue.tfdataset import dataset, dataset_eval, dataset_train
from stagedml.datasets.squad.tf import squad11_train_dataset
from stagedml.models.bert import ( BertLayer )
from stagedml.models.bert_squad import BertSquadLogitsLayer
from stagedml.utils.refs import ( Squad11TFR, BertSquad )
from stagedml.utils.tf import ( KerasBuild, protocol_add, dpurge, keras_save,
    match_metric )

from typing import Optional,Any,List,Tuple,Union


def config(tfrecs:Squad11TFR)->Config:
  name = 'bert-finetune-squad'

  train_tfrecord_refpath = [tfrecs, 'train.tfrecord']
  eval_tfrecord_refpath = [tfrecs, 'eval.tfrecord']
  task_config_refpath = [tfrecs, 'meta.json']

  bert_ckpt_refpath = store_cattrs(tfrecs).bert_ckpt_refpath
  bert_config_refpath = store_cattrs(tfrecs).bert_config

  lr = 2e-5
  train_epoches = 3
  train_batch_size = 8
  eval_batch_size = store_cattrs(tfrecs).eval_batch_size
  max_seq_length = store_cattrs(tfrecs).max_seq_length
  version=2
  return Config(locals())

class Model(KerasBuild):
  model_core:tf.keras.Model
  strategy:Any
  optimizer:Any


def build(m:Model, clear_session:bool=True)->None:
  tf.keras.backend.clear_session()
  c = build_cattrs(m)


  with open(build_path(m, c.bert_config_refpath), "r") as f:
    bert_config = BertConfig.from_dict(json_load(f))

  with open(build_path(m, c.task_config_refpath), "r") as f:
    task_config = json_load(f)

  c.train_data_size = task_config['train_data_size']
  c.eval_data_size = task_config['eval_data_size']
  c.train_steps_per_epoch = int(c.train_data_size / c.train_batch_size)
  c.eval_steps_per_epoch = int(c.eval_data_size / c.eval_batch_size)
  c.train_warmup_steps = int(c.train_epoches * c.train_data_size * 0.1 / c.train_batch_size)

  m.strategy=tf.distribute.MirroredStrategy()
  with m.strategy.scope():
    input_word_ids = tf.keras.Input(shape=(c.max_seq_length,), name='input_word_ids', dtype=tf.int32)
    input_mask     = tf.keras.Input(shape=(c.max_seq_length,), name='input_mask', dtype=tf.int32)
    input_type_ids = tf.keras.Input(shape=(c.max_seq_length,), name='input_type_ids', dtype=tf.int32)
    inputs = {
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids
        }

    bert = BertLayer(config=bert_config, float_type=tf.float32, name='bert')
    bert_outputs = bert(input_word_ids, input_mask, input_type_ids)
    bert_model = tf.keras.Model(inputs=inputs, outputs=bert_outputs)
    bert_model.summary()

    _,_,_,hidden_outputs = bert_model(inputs)

    squad_logits_layer = BertSquadLogitsLayer(
        initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range))
    start_logits, end_logits = squad_logits_layer(hidden_outputs[-1])

    model = tf.keras.Model(inputs=inputs, outputs=[start_logits, end_logits], name='squad_model')
    optimizer = create_optimizer(c.lr, c.train_steps_per_epoch*c.train_epoches, c.train_warmup_steps)

    m.model = model
    m.model_core = bert_model
    m.optimizer = optimizer
  return


# def cploaded(s:State)->State:
#   return state_add(s,'cpload')
def cpload(m:Model)->None:
  """ Load checkpoint into model """
  c = build_cattrs(m)
  checkpoint = tf.train.Checkpoint(model=m.model_core)
  checkpoint.restore(build_path(m, c.bert_ckpt_refpath)).assert_consumed()
  protocol_add(m, 'cpload')


# def ctrained(s:State)->State:
#   return state_add(s, 'ctrain')
def ctrain(m:Model)->None:
  assert m.model is not None

  c = build_cattrs(m)
  o = build_outpath(m)
  print('Training')

  def _loss_fn(labels, model_outputs):
    loss_factor = 1.0
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    start_logits, end_logits = model_outputs

    start_loss = tf.keras.backend.sparse_categorical_crossentropy(
        start_positions, start_logits, from_logits=True)
    end_loss = tf.keras.backend.sparse_categorical_crossentropy(
        end_positions, end_logits, from_logits=True)

    total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
    total_loss *= loss_factor
    return total_loss


  # def _metric_fn():
  #   return \
  #     tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy', dtype=tf.float32)

  def _get_model():
    return m.model,None

  def _train_input_fn()->Any:
    return squad11_train_dataset(
      input_file=build_path(m, c.train_tfrecord_refpath),
      max_seq_length=c.max_seq_length,
      train_batch_size=c.train_batch_size)

  def _eval_input_fn()->Any:
    assert False

  tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
                                     write_graph=False, update_freq='batch')
  m.model.optimizer = m.optimizer
  logging.set_verbosity(logging.INFO)
  run_customized_training_loop(
      strategy=m.strategy,
      model_fn=_get_model,
      loss_fn=_loss_fn,
      model_dir=o,
      steps_per_epoch=c.train_steps_per_epoch,
      steps_per_loop=2*c.train_batch_size,
      epochs=c.train_epoches,
      train_input_fn=_train_input_fn,
      # eval_input_fn=_eval_input_fn,
      # eval_steps=c.valid_steps_per_epoch,
      init_checkpoint=None,
      # metric_fn=_metric_fn,
      custom_callbacks=None,
      run_eagerly=False)

  dpurge(o,'ctl_step.*ckpt', debug=True)
  with open(join(o,'summaries/training_summary.txt'), 'r') as f:
    s=json_load(f)
  protocol_add(m, 'ctrain', result=s)
  return



# def evaluated(s:State)->State:
#   return state_add(s, 'evaluate')
def evaluate(m:Model)->Model:
  assert False, 'TODO'
  # protocol_add_eval(m, 'evaluate', k.metrics_names, h)
  return m


def bert_finetune_squad11(m:Manager, *args, **kwargs)->BertSquad:
  def _realize(dref:DRef,context:Context)->List[Path]:
    b=Model(mkbuild(dref,context));
    build(b); cpload(b); ctrain(b);
    # FIXME: evaluate(b);
    keras_save(b)
    return [build_outpath(b)]
  return BertSquad(mkdrv(m,
    config=config(*args, **kwargs),
    matcher=match_latest(), #match_metric('evaluate', 'eval_accuracy'),
    realizer=_realize))

