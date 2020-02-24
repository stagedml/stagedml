import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from pylightnix import ( Path, Config, Manager, RRef, DRef, Context,
    store_cattrs, build_path, build_outpath, build_cattrs, mkdrv, rref2path,
    json_load, build_config, mkconfig, mkbuild, match_only, build_wrapper_,
    tryread )

from stagedml.imports import ( join, clear_session, set_session_config,
    TensorBoard, ModelCheckpoint )

from stagedml.utils.tf import ( runtb, runtensorboard, thash, KerasBuild,
    protocol_add, protocol_add_hist, protocol_add_eval, match_metric, dpurge,
    keras_save, keras_wrapper )
from stagedml.models.transformer import ( Transformer, create_train_model,
    create_optimizer, train_ds, LearningRateScheduler, LearningRateFn,
    BASE_PARAMS )

from stagedml.types import ( WmtTfrecs, TransWmt, Dict )

from typing import Optional,Any,List,Tuple,Union


from official.utils.flags._performance import DTYPE_MAP

def config(wmt:WmtTfrecs):
  name = 'transformer_wmt'
  version = 1
  enable_xla = False
  num_gpus = 1
  data_dir = [ wmt ]
  vocab_refpath = [ wmt, store_cattrs(wmt).vocab_file ]
  dtype = 'fp32'
  train_steps = 300000
  steps_between_evals = 1000

  params:Dict[str,Any] = dict(BASE_PARAMS)
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


class TransformerBuild(KerasBuild):
  params:dict


def build(b:TransformerBuild)->None:
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

  b.model = create_train_model(Transformer, c.params)
  b.model.compile(create_optimizer(c.params))
  b.model.summary()


def train(b:TransformerBuild):
  c = build_cattrs(b)
  o = build_outpath(b)

  callbacks = [
      TensorBoard(log_dir=c.params["model_dir"]),
      LearningRateScheduler(
          LearningRateFn(c.params["learning_rate"],
                         c.params["hidden_size"],
                         c.params["learning_rate_warmup_steps"]), 0),
      ModelCheckpoint(
        filepath=join(o, "checkpoint.ckpt"),
        save_weights_only=True,
        verbose=True)]

  num_epoches = c.train_steps // c.steps_between_evals
  print(num_epoches)

  history = b.model.fit(
    train_ds(c.params),
    initial_epoch=0,
    epochs=num_epoches,
    steps_per_epoch=c.steps_between_evals,
    callbacks=callbacks,
    verbose=True)

    # current_step=0
    # while current_step < c.train_steps:
    #   current_iteration = current_step // c.steps_between_evals
    #   remaining_steps = c.train_steps - current_step
    #   train_steps_per_eval = (remaining_steps if remaining_steps < c.steps_between_evals
    #                                           else c.steps_between_evals )
    #   history = model.fit(
    #       train_ds(c.params),
    #       initial_epoch=current_iteration,
    #       epochs=current_iteration + 1,
    #       steps_per_epoch=train_steps_per_eval,
    #       callbacks=callbacks,
    #       validation_split=0.2,
    #       verbose=False)
    #   current_step += train_steps_per_eval
    #   print("Train history: {}".format(history.history))
    # print("End train iteration at global step:{}".format(current_step))

def realize(b:TransformerBuild)->None:
  build(b)
  train(b)

def transformer_wmt(m:Manager, wmt:WmtTfrecs)->TransWmt:
  """ FIXME: set appropriate matcher """
  return TransWmt(mkdrv(m, config=config(wmt),
                           matcher=match_only(),
                           realizer=build_wrapper_(realize, TransformerBuild)))

