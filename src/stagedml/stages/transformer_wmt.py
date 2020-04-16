import tensorflow as tf
assert tf.version.VERSION.startswith('2.1') or \
       tf.version.VERSION.startswith('2.2')

from pylightnix import ( Build, Path, Config, Manager, RRef, DRef, Context,
    store_cattrs, build_path, build_outpaths, build_cattrs, mkdrv, rref2path,
    json_load, build_config, mkconfig, mkbuild, match_only, build_wrapper_,
    tryread, store_config, mklens, repl_realize, instantiate, shell, promise,
    claim, repl_build, build_context, match_best, build_outpaths,
    build_setoutpaths )

from stagedml.imports import ( join, clear_session, set_session_config,
    TensorBoard, ModelCheckpoint, copy_tree, Model, isfile, get_single_element,
    deepcopy, copyfile, SummaryWriter, create_file_writer )

from stagedml.utils import ( runtb, thash, dpurge, tryindex )

from stagedml.models.transformer import ( create_train_model, create_eval_model,
    create_optimizer, train_ds, eval_ds, LearningRateScheduler, LearningRateFn,
    BASE_PARAMS, Subtokenizer, EOS_ID, bleu_wrapper)

from stagedml.models.transformer.data import eval_ds, predict_ds

from stagedml.types import ( WmtSubtok, TransWmt, Dict, Optional,Any,List,Tuple,Union )

from stagedml.stages.fetchwmt import create_subtokenizer

from official.utils.flags._performance import DTYPE_MAP

class TransformerBuild(Build):
  params:dict
  model:Model
  eval_model:Model
  train_model:Model
  epoch:Optional[int]
  filewriter:SummaryWriter
  tbcallback:TensorBoard
  subtokenizer:Subtokenizer

# def cont(b:TransformerBuild)->None:
#   copy_tree(rref2path(build_cattrs(b).continue_from),build_outpath(b))

def compute_missing_params(b, instance_idx:int):
  c = build_cattrs(b)
  o = build_outpaths(b)[instance_idx]
  me = mklens(b,o)
  c.params["model_dir"] = o
  c.params["dtype"] = DTYPE_MAP[c.dtype]
  c.params["data_dir"] = me.data_dir.syspath
  vocab_contents = tryread(me.vocab_refpath.syspath)
  assert vocab_contents is not None
  c.params["vocab_size"] = len(vocab_contents.split('\n'))
  # print(f'Setting vocab_size to {c.params["vocab_size"]}')


def build(b:TransformerBuild, instance_idx:int)->None:
  c = build_cattrs(b)
  o = build_outpaths(b)[instance_idx]
  me = mklens(b,o)

  clear_session()
  compute_missing_params(b, instance_idx)
  set_session_config(enable_xla=c.enable_xla)

  b.train_model = create_train_model(c.params)
  b.train_model.compile(create_optimizer(c.params))
  # b.train_model.summary()

  b.eval_model = create_eval_model(c.params)
  # b.eval_model.summary()
  b.epoch = None
  b.filewriter = create_file_writer(join(o,'eval'))
  b.tbcallback = TensorBoard(log_dir=o, profile_batch=0, write_graph=False)
  b.subtokenizer = create_subtokenizer(WmtSubtok(me.wmt.dref), build_context(b))


def loadcp(b:TransformerBuild, instance_idx):
  me = mklens(b,o=build_outpaths(b)[instance_idx])
  ckpt0 = me.checkpoint_init.val
  assert ckpt0 is not None
  b.train_model.load_weights(ckpt0)
  b.epoch = None


def evaluate(b:TransformerBuild, instance_idx:int)->None:
  assert b.train_model is not None
  c = build_cattrs(b)
  me = mklens(b,o=build_outpaths(b)[instance_idx])
  input_txt:Path=me.eval_input_refpath.syspath
  target_src_txt:Path=me.eval_target_refpath.syspath
  # print('Evaluating')
  epoch=b.epoch if b.epoch is not None else 0
  ds = eval_ds(b.subtokenizer,
               input_txt,
               target_src_txt,
               batch_size=me.eval_batch_size.val,
               params=me.params.val,
               take=me.eval_steps.val)
  h=b.train_model.evaluate(ds, verbose=1, callbacks=[b.tbcallback])
  with b.filewriter.as_default():
    for mname,v in zip(b.train_model.metrics_names,h):
      tf.summary.scalar('epoch_'+mname,v,step=epoch)


def predict(b:TransformerBuild, instance_idx:int)->None:
  assert b.eval_model is not None
  c = build_cattrs(b)
  o = build_outpaths(b)[instance_idx]
  me = mklens(b,o=o)
  input_txt:Path=me.eval_input_refpath.syspath
  target_src_txt:Path=me.eval_target_refpath.syspath
  target_txt=join(o,'targets.txt')
  epoch=b.epoch if b.epoch is not None else 0
  output_txt:Path=Path(join(o,f"output-{epoch}.txt"))
  b.eval_model.load_weights(me.checkpoint_refpath.syspath)

  print('Predicting')
  ds = predict_ds(b.subtokenizer,
                  input_txt,
                  batch_size=me.eval_batch_size.val,
                  params=me.params.val,
                  take=me.eval_steps.val)

  outputs,_ = b.eval_model.predict(ds, verbose=1, callbacks=[b.tbcallback])
  with open(output_txt,'w') as fhyp, \
       open(target_src_txt,'r') as ftgt_src, \
       open(target_txt,'w') as ftgt:
    neols=0
    for ids in outputs:
      reply=' '.join(b.subtokenizer.decode(ids[:tryindex(list(ids),EOS_ID)]).splitlines())
      target=ftgt_src.readline().strip()
      fhyp.write(reply); fhyp.write('\n')
      ftgt.write(target); ftgt.write('\n')

  bleu_uncased = bleu_wrapper(target_txt, output_txt, False)
  bleu_cased = bleu_wrapper(target_txt, output_txt, True)
  print(f'BLEU (uncased): {bleu_uncased}\n'
        f'BLEU (cased): {bleu_cased}\n')
  with b.filewriter.as_default():
    tf.summary.scalar('bleu_cased',bleu_cased,step=epoch)
    tf.summary.scalar('bleu_uncased',bleu_uncased,step=epoch)
  with open(me.bleu_refpath.syspath,'w') as f:
    f.write(f"{(bleu_cased + bleu_uncased)/2.0}\n")


def train(b:TransformerBuild, instance_idx:int=0):
  assert b.train_model is not None
  c = build_cattrs(b)
  o = build_outpaths(b)[instance_idx]
  me = mklens(b,o)
  callbacks = [
    b.tbcallback,
    LearningRateScheduler(
        LearningRateFn(c.params["learning_rate"],
                       c.params["hidden_size"],
                       c.params["learning_rate_warmup_steps"]), 0)
        ]

  nepoches = getattr(c,'hack_nepoch', c.train_steps // c.steps_between_evals)
  b.epoch = 0 if b.epoch is None else b.epoch
  while b.epoch < nepoches:
    history = b.train_model.fit(
      train_ds(c.params),
      initial_epoch=b.epoch,
      epochs=b.epoch+1,
      steps_per_epoch=c.steps_between_evals,
      callbacks=callbacks,
      verbose=True)

    ckpt = me.checkpoint_refpath.syspath
    print(f"Saving '{ckpt}'")
    b.train_model.save_weights(ckpt)
    evaluate(b, instance_idx)
    predict(b, instance_idx)
    b.epoch += 1


def transformer_wmt(m:Manager, wmt:WmtSubtok, num_instances:int=1)->TransWmt:
  train_steps = 300000

  def _realize(b:TransformerBuild)->None:
    build_setoutpaths(b,num_instances)
    for instance_idx in range(num_instances):
      # print(f'Building Transformer instance {instance_idx+1} of {num_instances}')
      build(b,instance_idx)
      train(b,instance_idx)

  def _config():
    name = 'transformer-nmt-'+mklens(wmt).name.val
    version = 1
    enable_xla = False
    num_gpus = 1
    data_dir = [wmt]
    vocab_refpath = mklens(wmt).vocab_file.refpath
    dtype = 'fp32'
    nonlocal train_steps
    steps_between_evals = 5000

    eval_steps:Optional[int] = None # all
    eval_batch_size = 30

    params:Dict[str,Any] = dict(BASE_PARAMS)
    params["num_gpus"] = num_gpus
    params["use_ctl"] = False
    params["static_batch"] = False
    params["max_length"] = 256
    params["decode_batch_size"] = 32
    params["decode_max_length"] = 128
    params["padded_decode"] = False
    params["num_parallel_calls"] = 4
    params["use_synthetic_data"] = False
    params["batch_size"] = params["default_batch_size"]
    params["repeat_dataset"] = None
    params["enable_metrics_in_training"] = True

    bleu_refpath = [promise, 'bleu.txt']
    checkpoint_refpath = [claim, 'checkpoint.ckpt']
    eval_input_refpath=mklens(wmt).eval_input_combined.refpath
    eval_target_refpath=mklens(wmt).eval_target_combined.refpath
    return mkconfig(locals())

  return TransWmt(mkdrv(m, config=_config(),
                           matcher=match_best('bleu.txt'),
                           realizer=build_wrapper_(_realize, TransformerBuild)))

