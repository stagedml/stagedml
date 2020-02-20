import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from pylightnix import ( Path, Config, Manager, RRef, DRef, Context,
    store_cattrs, build_path, build_outpath, build_cattrs, mkdrv, rref2path,
    json_load, build_config, mkconfig, mkbuild, match_only, build_wrapper,
    tryread )

from stagedml.imports import ( join, clear_session, set_session_config,
    TensorBoard, ModelCheckpoint )

from stagedml.utils.tf import ( runtb, runtensorboard, thash, KerasBuild,
    protocol_add, protocol_add_hist, protocol_add_eval, match_metric, dpurge,
    keras_save, keras_wrapper )
from stagedml.types import ( Wmt, TransWmt )
from stagedml.models.transformer import ( Transformer, create_train_model,
    create_optimizer, train_ds, LearningRateScheduler, LearningRateFn,
    BASE_PARAMS )

from typing import Optional,Any,List,Tuple,Union


from official.utils.flags._performance import DTYPE_MAP

def config(wmt:Wmt):
  name = 'transformer_wmt32ende'
  version = 1
  enable_xla = False
  num_gpus = 1
  data_dir = [ wmt ]
  vocab_refpath = [ wmt, 'vocab.ende.32768' ]
  dtype = 'fp32'
  train_steps = 300000
  steps_between_evals = 1000

  params = dict(BASE_PARAMS)
  params["num_gpus"] = num_gpus
  params["use_ctl"] = False
  # params["data_dir"] = flags_obj.data_dir
  # params["model_dir"] = flags_obj.model_dir
  params["static_batch"] = False
  params["max_length"] = 256
  params["decode_batch_size"] = 32
  params["decode_max_length"] = 128
  params["padded_decode"] = False
  params["num_parallel_calls"] = 4
  params["use_synthetic_data"] = False
  params["batch_size"] = 512
  params["repeat_dataset"] = None
  params["enable_metrics_in_training"] = False

  return mkconfig(locals())

def build(b:KerasBuild)->None:
  c = build_cattrs(b)
  o = build_outpath(b)

  c.params["dtype"] = DTYPE_MAP[c.dtype]
  c.params["data_dir"] = build_path(b, c.data_dir)
  c.params["model_dir"] = o
  vocab_contents = tryread(build_path(b, c.vocab_refpath))
  assert vocab_contents is not None
  c.params["vocab_size"] = len(vocab_contents.split('\n'))
  print(f'Setting vocab_size to {c.params["vocab_size"]}')

  clear_session()
  set_session_config(enable_xla=c.enable_xla)

  with tf.distribute.MirroredStrategy().scope():

    model = create_train_model(Transformer, c.params)
    opt = create_optimizer(c.params)

    # summary_writer = tf.compat.v2.summary.create_file_writer(c.params['model_dir'])
    model.compile(opt)
    model.summary()

    callbacks = [
        TensorBoard(log_dir=c.params["model_dir"]),
        LearningRateScheduler(
            LearningRateFn(c.params["learning_rate"],
                           c.params["hidden_size"],
                           c.params["learning_rate_warmup_steps"]), 0),
        ModelCheckpoint(join(o, "cp-{epoch:04d}.ckpt"), save_weights_only=True)]


    current_step=0
    while current_step < c.train_steps:
      current_iteration = current_step // c.steps_between_evals
      remaining_steps = c.train_steps - current_step
      train_steps_per_eval = (remaining_steps if remaining_steps < c.steps_between_evals
                                              else c.steps_between_evals )
      history = model.fit(
          train_ds(c.params),
          initial_epoch=current_iteration,
          epochs=current_iteration + 1,
          steps_per_epoch=train_steps_per_eval,
          callbacks=callbacks,
          verbose=False)
      current_step += train_steps_per_eval
      print("Train history: {}".format(history.history))

    print("End train iteration at global step:{}".format(current_step))


def transformer_wmt32ende(m:Manager, wmt:Wmt)->TransWmt:
  """ FIXME: set appropriate matcher """
  def _realize(b:KerasBuild)->None:
    build(b)
  return TransWmt(mkdrv(m, config=config(wmt),
                           matcher=match_only(),
                           realizer=keras_wrapper(build)))
