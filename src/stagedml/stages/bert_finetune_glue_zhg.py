import tensorflow as tf
assert tf.version.VERSION.startswith('2.1') or \
       tf.version.VERSION.startswith('2.2')

from stagedml.datasets.glue.tfdataset import (dataset_test, dataset_train,
                                              dataset_valid,
                                              bert_finetune_dataset )

from stagedml.imports.sys import (join, partial, Build, Path, Config,
                                  Manager, RRef, DRef, Context, store_cattrs,
                                  build_outpaths, build_cattrs, mkdrv,
                                  rref2path, json_load, build_config, mklens,
                                  build_wrapper_, mkconfig, promise, claim,
                                  build_setoutpaths, realize, instantiate,
                                  repl_realize)

from stagedml.core import (protocol_add, protocol_add_hist, protocol_add_eval,
                           protocol_match, dpurge )

from stagedml.types import (BertCP, GlueTFR, BertGlue,
                            Optional,Any,List,Tuple,Union,BertFinetuneTFR)


from keras_bert import (Tokenizer, get_base_dict, get_model, compile_model,
                        gen_batch_inputs, extract_embeddings, POOL_NSP,
                        POOL_MAX, get_pretrained, PretrainedList,
                        get_checkpoint_paths)

from stagedml.imports.tf import (Model, clear_session)

from stagedml.utils.tf import (runtb, runtensorboard, thash, modelhash,
                               print_model_checkpoint_diff, SparseF1Score,
                               dataset_cardinality_size, dataset_iter_size)

try:
  from ipdb import set_trace
except ModuleNotFoundError:
  pass

class State(Build):
  model:Model



def build(s:State):
  """ Build the model """
  clear_session()

  with open(mklens(s).refbert.bert_vocab.syspath) as f:
    vocab = {text:val for val,text in enumerate(f.read().split())}

  set_trace()

  model = get_model(
    token_num=len(vocab.keys()),
    head_num=mklens(s).bert_config.num_attention_heads.val,
    transformer_num=mklens(s).bert_config.num_hidden_layers.val,
    embed_dim=mklens(s).bert_config.hidden_size.val,
    feed_forward_dim=mklens(s).bert_config.intermediate_size.val,
    pos_num=mklens(s).bert_config.max_position_embeddings.val,
    seq_len=mklens(s).max_seq_length.val,
    dropout_rate=0.05,
  )
  compile_model(model)
  model.summary()
  s.model = model


def cpload(b:Model, iid:int=0)->None:
  """ Load checkpoint into model """
  pass



def train(b:Model, iid:int=0)->None:
  """ Train the model by using """
  pass


def evaluate(b:Model, iid:int=0)->None:
  """ Evaluate the model """
  pass


def bert_finetune_glue_zhg(m:Manager, refbert:BertCP, tfrecs:BertFinetuneTFR,
                           num_instances:int=1)->BertGlue:

  def _config()->dict:
    name = 'bert-finetune-'+mklens(tfrecs).task_name.val.lower()
    datasets={
      'train':mklens(tfrecs).outputs.train.refpath,
      'valid':mklens(tfrecs).outputs.valid.refpath,
      'test':mklens(tfrecs).outputs.test.refpath}
    bert_config = mklens(refbert).bert_config.refpath
    bert_ckpt = mklens(refbert).bert_ckpt.refpath
    assert mklens(refbert).bert_vocab.refpath==\
           mklens(tfrecs).bert_vocab.refpath, \
      "Model dictionary path doesn't match the dataset dictionary path"
    num_classes=mklens(tfrecs).num_classes.val
    max_seq_length=mklens(tfrecs).max_seq_length.val
    lr = 2e-5
    train_batch_size = 32
    test_batch_size = 32
    train_epoches = 3

    out_ckpt = [claim, f'{name}.ckpt']
    out_protocol = [promise, 'protocol.json']
    return locals()

  def _make(b:Model)->None:
    build_setoutpaths(b,num_instances)
    for i in range(num_instances):
      build(b);
      if mklens(b).bert_ckpt_in.optval is not None:
        cpload(b,i)
      train(b,i)
      evaluate(b,i)

  return BertGlue(mkdrv(m,
    config=mkconfig(_config()),
    matcher=protocol_match('evaluate', 'eval_accuracy'),
    realizer=build_wrapper_(_make, State)))

