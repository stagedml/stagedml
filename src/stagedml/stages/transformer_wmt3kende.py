import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from pylightnix import ( Path, Config, Manager, RRef, DRef, Context,
    store_cattrs, build_path, build_outpath, build_cattrs, mkdrv, rref2path,
    json_load, build_config, mkconfig, mkbuild, match_only, build_wrapper )

from stagedml.imports import ( join )
from stagedml.imports_tf import ( clear_session, set_session_config,
    TensorBoard, ModelCheckpoint )

from stagedml.utils.tf import ( runtb, runtensorboard, thash, KerasBuild,
    protocol_add, protocol_add_hist, protocol_add_eval, match_metric, dpurge,
    keras_save, keras_wrapper )
from stagedml.utils.refs import ( Wmt, TransWmt )
from stagedml.models.transformer import ( Transformer, create_train_model,
    create_optimizer, train_ds, LearningRateScheduler, LearningRateFn )

from typing import Optional,Any,List,Tuple,Union


from official.utils.flags._performance import DTYPE_MAP

def config(wmt:Wmt):
  name = 'transformer_wmt3kende'
  version = 1
  enable_xla = True
  num_gpus = 1
  data_dir = [ wmt ]
  dtype = 'fp32'
  train_steps = 300000

  params:dict = {}
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
  params["batch_size"] = 32
  params["repeat_dataset"] = None
  params["enable_metrics_in_training"] = True
  params["steps_between_evals"] = 1000

  return mkconfig(locals())

def build(b:KerasBuild)->None:
  c = build_cattrs(b)
  o = build_outpath(b)

  c.params["dtype"] = DTYPE_MAP[c.dtype]
  c.params["data_dir"] = build_path(b, c.data_dir)
  c.params["model_dir"] = o

  clear_session()
  set_session_config(enable_xla=c.enable_xla)

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
        initial_epoch=0,
        epochs=current_iteration + 1,
        steps_per_epoch=c.train_steps_per_eval,
        callbacks=callbacks,
        verbose=False)
    current_step += train_steps_per_eval
    print("Train history: {}".format(history.history))

  print("End train iteration at global step:{}".format(current_step))


def transformer_wmt3kende(m:Manager, wmt:Wmt)->TransWmt:
  """ FIXME: set appropriate matcher """
  def _realize(b:KerasBuild)->None:
    build(b)
  return TransWmt(mkdrv(m, config=config(wmt),
                           matcher=match_only(),
                           realizer=keras_wrapper(build)))
