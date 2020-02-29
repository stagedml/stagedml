import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from pylightnix import ( Path, Config, Manager, RRef, DRef, Context,
    store_cattrs, build_path, build_outpath, build_cattrs, mkdrv, rref2path,
    json_load, build_config, mkconfig, mkbuild, match_only, build_wrapper_,
    tryread, store_config, mklens )

from stagedml.imports import ( join, clear_session, set_session_config,
    TensorBoard, ModelCheckpoint, copy_tree, Model, isfile, get_single_element,
    deepcopy )

from stagedml.utils import ( runtb, runtensorboard, thash, KerasBuild,
    protocol_add, protocol_add_hist, protocol_add_eval, match_metric, dpurge,
    keras_save, keras_wrapper, tryindex )

from stagedml.models.transformer import ( create_train_model, create_eval_model,
    create_optimizer, train_ds, eval_ds, LearningRateScheduler, LearningRateFn,
    BASE_PARAMS, Subtokenizer, EOS_ID )

from stagedml.types import ( WmtTfrecs, TransWmt, Dict, Optional,Any,List,Tuple,Union )


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
  params["batch_size"] = params["default_batch_size"]
  eval_batch_size = 10
  params["repeat_dataset"] = None
  params["enable_metrics_in_training"] = True

  # continue_from = 'rref:81a6fd23c0086b3968428f782c89d39d-74c441ccbe5bede1521db7d2a7644ca1-transformer_wmt'
  # hack_nepoch = 1
  # hack_checkpoint = "/workspace/_pylightnix/tmp/200226-14:45:52:334295+0300_f807db56_t4h0dsiw/checkpoint.ckpt"
  hack_checkpoint = \
    join(rref2path(RRef(("rref:dff59d4cc6d49f67e9e5edfec8359d13-"
                         "f807db56d12ef3f79298b51988c47a05-"
                         "transformer_wmt"))),"checkpoint.ckpt")
  return mkconfig(locals())

class TransformerBuild(KerasBuild):
  params:dict
  eval_model:Model
  train_model:Model

def cont(b:TransformerBuild)->None:
  copy_tree(rref2path(build_cattrs(b).continue_from),build_outpath(b))

def compute_missing_params(b):
  c = build_cattrs(b)
  o = build_outpath(b)
  c.params["dtype"] = DTYPE_MAP[c.dtype]
  c.params["data_dir"] = build_path(b, c.data_dir)
  c.params["model_dir"] = o
  vocab_contents = tryread(build_path(b, c.vocab_refpath))
  assert vocab_contents is not None
  c.params["vocab_size"] = len(vocab_contents.split('\n'))
  print(f'Setting vocab_size to {c.params["vocab_size"]}')


def build(b:TransformerBuild, mode:str)->None:
  c = build_cattrs(b)
  o = build_outpath(b)

  clear_session()
  compute_missing_params(b)
  set_session_config(enable_xla=c.enable_xla)

  if mode=='train':
    b.train_model = create_train_model(c.params)
    b.train_model.compile(create_optimizer(c.params))
    b.train_model.summary()
  elif mode=='eval':
    b.eval_model = create_eval_model(c.params)
    b.eval_model.compile()
    checkpoint = getattr(c,'hack_checkpoint', join(o,'checkpoint.ckpt'))
    print(f'Loading checkpoint {c}...')
    b.eval_model.load_weights(checkpoint)
    b.eval_model.summary()
  else:
    assert False, f"Invalid build mode '{mode}'"

def build_train(b:TransformerBuild):
  return build(b,'train')
def build_eval(b:TransformerBuild):
  return build(b,'eval')


def evaluate(b:TransformerBuild)->None:
  assert b.eval_model is not None
  o = build_outpath(b)
  c = build_cattrs(b)
  subtokenizer = Subtokenizer(build_path(b, c.vocab_refpath))
  ds = eval_ds(subtokenizer,
               mklens(b).wmt.eval_input_combined.syspath,
               batch_size=c.eval_batch_size,
               params=c.params)
  # ds = ds.take(2) # FIXME
  outputs,_ = b.eval_model.predict(ds, verbose=1)
  with open(join(o,'translation.txt'),'w') as f:
    neols=0
    for ids in outputs:
      reply=subtokenizer.decode(ids[:tryindex(list(ids),EOS_ID)])
      if '\n' in reply:
        neols+=1
        reply=' '.join(reply.split('\n'))
      f.write(reply)
      f.write('\n')
    print(f'Spurious EOLs seen: {neols}')


def train(b:TransformerBuild):
  assert b.train_model is not None
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

  num_epoches = getattr(c,'hack_nepoch', c.train_steps // c.steps_between_evals)
  print(num_epoches)

  history = b.train_model.fit(
    train_ds(c.params),
    initial_epoch=0,
    epochs=num_epoches,
    steps_per_epoch=c.steps_between_evals,
    callbacks=callbacks,
    verbose=True)


def _realize(b:TransformerBuild)->None:
  build_train(b)
  train(b)

def transformer_wmt(m:Manager, wmt:WmtTfrecs)->TransWmt:
  """ FIXME: set appropriate matcher """
  return TransWmt(mkdrv(m, config=config(wmt),
                           matcher=match_only(),
                           realizer=build_wrapper_(_realize, TransformerBuild)))

