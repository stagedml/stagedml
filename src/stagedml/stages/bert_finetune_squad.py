import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from os.path import join
from absl import logging
from tensorflow.python.keras.backend import clear_session
from tensorflow.keras.callbacks import TensorBoard

from official.nlp.bert.configs import BertConfig
from official.nlp.optimization import create_optimizer
from official.modeling.model_training_utils import run_customized_training_loop

from pylightnix import ( Build, Path, Manager, Config, DRef, RRef, Context,
    match_only, store_cattrs, build_cattrs, build_outpath, json_load, mkbuild,
    mkdrv, match_latest, mklens, build_wrapper_, promise, mkconfig )

from stagedml.datasets.squad.tfrecord import tf_record_dataset
from stagedml.models.bert import ( BertLayer, BertInput, BertOutput, BertModel )
from stagedml.models.bert_squad import BertSquadLogitsLayer
from stagedml.types import ( Squad11TFR, BertSquad, Optional, Any, List, Tuple, Union )
from stagedml.utils import ( dpurge, modelhash )
from stagedml.core import ( protocol_add, protocol_match )


class Model(Build):
  model:tf.keras.Model
  model_core:tf.keras.Model
  strategy:Any
  optimizer:Any


def build(m:Model, clear_session:bool=True)->None:
  tf.keras.backend.clear_session()
  c = build_cattrs(m)

  with open(mklens(m).bert_config.syspath, "r") as f:
    bert_config = BertConfig.from_dict(json_load(f))

  with open(mklens(m).task_config.syspath, "r") as f:
    task_config = json_load(f)

  c.train_data_size = task_config['train_data_size']
  c.eval_data_size = task_config['eval_data_size']
  c.train_steps_per_epoch = int(c.train_data_size / c.train_batch_size)
  c.eval_steps_per_epoch = int(c.eval_data_size / c.eval_batch_size)
  c.train_warmup_steps = int(c.train_epoches * c.train_data_size * 0.1 / c.train_batch_size)

  m.strategy=tf.distribute.MirroredStrategy()
  with m.strategy.scope():
    bert_inputs = BertInput(
        tf.keras.Input(shape=(c.max_seq_length,), name='input_word_ids', dtype=tf.int32),
        tf.keras.Input(shape=(c.max_seq_length,), name='input_mask', dtype=tf.int32),
        tf.keras.Input(shape=(c.max_seq_length,), name='input_type_ids', dtype=tf.int32))

    bert = BertLayer(config=bert_config, float_type=tf.float32, name='bert')
    bert_model = BertModel(bert_inputs, bert(bert_inputs))
    bert_model.model.summary()
    bert_outputs = bert_model(bert_inputs)

    squad_logits_layer = BertSquadLogitsLayer(
        initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range))
    start_logits, end_logits = squad_logits_layer(bert_outputs.hidden_output[-1])

    model = tf.keras.Model(inputs=bert_inputs, outputs=[start_logits, end_logits], name='squad_model')
    optimizer = create_optimizer(c.lr, c.train_steps_per_epoch*c.train_epoches, c.train_warmup_steps)

    m.model = model
    m.model_core = bert_model.model
    m.optimizer = optimizer
  return


# def cploaded(s:State)->State:
#   return state_add(s,'cpload')
def cpload(m:Model)->None:
  """ Load checkpoint into model """
  c = build_cattrs(m)
  checkpoint = tf.train.Checkpoint(model=m.model_core)
  checkpoint.restore(mklens(m).bert_ckpt.syspath).assert_consumed()
  protocol_add(mklens(m).protocol.syspath, 'cpload', whash=modelhash(m.model))


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

  def _train_input_fn(ctx:Any)->Any:
    global_batch_size=c.train_batch_size
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    return tf_record_dataset(
      input_file=mklens(m).train_tfrecord.syspath,
      max_seq_length=c.max_seq_length,
      train_batch_size=batch_size)

  def _eval_input_fn()->Any:
    assert False

  tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
                                     write_graph=False, update_freq='batch')
  m.model.optimizer = m.optimizer
  logging.set_verbosity(logging.INFO)
  tm=run_customized_training_loop(
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

  protocol_add(mklens(m).protocol.syspath, 'ctrain', whash=modelhash(m.model), result=s)
  # FIXME: save checkpoint?



# def evaluated(s:State)->State:
#   return state_add(s, 'evaluate')
def evaluate(m:Model)->Model:
  assert False, 'TODO'
  # protocol_add_eval(m, 'evaluate', k.metrics_names, h)
  return m

def _realize(b:Model)->None:
  build(b); cpload(b); ctrain(b);

def bert_finetune_squad11(m:Manager, tfrecs:Squad11TFR)->BertSquad:

  def _config()->dict:
    name = 'bert-finetune-squad'

    train_tfrecord = mklens(tfrecs).output_train.refpath
    eval_tfrecord = mklens(tfrecs).output_eval.refpath
    task_config = mklens(tfrecs).output_meta.refpath

    bert_ckpt = mklens(tfrecs).bertref.bert_ckpt.refpath
    bert_config = mklens(tfrecs).bertref.bert_config.refpath

    lr = 2e-5
    train_epoches = 3
    train_batch_size = 8
    eval_batch_size = store_cattrs(tfrecs).eval_batch_size
    max_seq_length = store_cattrs(tfrecs).max_seq_length

    protocol = [promise, 'protocol.json']
    broken = True
    version=4
    return locals()

  return BertSquad(mkdrv(m,
    config=mkconfig(_config()),
    matcher=match_latest(), #match_metric('evaluate', 'eval_accuracy'),
    realizer=build_wrapper_(_realize, Model)))

