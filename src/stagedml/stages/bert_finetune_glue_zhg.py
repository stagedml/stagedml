import tensorflow as tf
assert tf.version.VERSION.startswith('2.1') or \
       tf.version.VERSION.startswith('2.2')

from stagedml.datasets.glue.tfdataset import (dataset_test, dataset_train,
                                              dataset_valid,
                                              bert_finetune_dataset )

from stagedml.imports.sys import (join, partial, Build, Path, Config,
                                  Manager, RRef, DRef, Context, store_cattrs,
                                  build_outpaths, build_cattrs, mkdrv,
                                  rref2path, readjson, build_config, mklens,
                                  build_wrapper_, mkconfig, promise, claim,
                                  build_setoutpaths, realize, instantiate,
                                  repl_realize, set_trace)

from stagedml.core import (protocol_add, protocol_add_hist, protocol_add_eval,
                           protocol_match, dpurge )

from stagedml.types import (BertCP, GlueTFR, BertGlue,
                            Optional,Any,List,Tuple,Union,BertFinetuneTFR)

from stagedml.imports.tf import (Model, clear_session, TFRecordDataset,
                                 TensorBoard, Dense)

from stagedml.utils.tf import (runtb, runtensorboard, thash, modelhash,
                               print_model_checkpoint_diff, SparseF1Score,
                               dataset_cardinality_size, dataset_iter_size)

from keras_bert import (Tokenizer, get_base_dict, get_model, compile_model,
                        gen_batch_inputs, extract_embeddings, POOL_NSP,
                        POOL_MAX, get_pretrained, PretrainedList,
                        get_checkpoint_paths,
                        load_model_weights_from_checkpoint)

from keras_radam import RAdam










class State(Build):
  model_pretrained:Model
  model_cls:Model
  config:dict



def build(s:State):
  """ Build the model """
  clear_session()

  with open(mklens(s).refbert.bert_vocab.syspath) as f:
    vocab = {text:val for val,text in enumerate(f.read().split())}

  model_pretrained = get_model(
    token_num=len(vocab.keys()),
    head_num=mklens(s).bert_config.num_attention_heads.val,
    transformer_num=mklens(s).bert_config.num_hidden_layers.val,
    embed_dim=mklens(s).bert_config.hidden_size.val,
    feed_forward_dim=mklens(s).bert_config.intermediate_size.val,
    pos_num=mklens(s).bert_config.max_position_embeddings.val,
    seq_len=mklens(s).max_seq_length.val,
    dropout_rate=0.05,
    training=True
  )

  inputs = model_pretrained.inputs[:2]
  outputs = Dense(
    mklens(s).num_classes.val, activation='softmax')(
      model_pretrained.get_layer('NSP-Dense').output)
  model_cls = Model(inputs, outputs)
  model_cls.summary()

  s.config = readjson(mklens(s).bert_config.syspath)
  s.model_pretrained = model_pretrained
  s.model_cls = model_cls


def cpload(s:State, iid:int=0)->None:
  """ Load checkpoint into model """
  l = mklens(s, build_output_idx=iid)
  ckpt = l.bert_ckpt.syspath
  load_model_weights_from_checkpoint(
    s.model_pretrained, s.config, ckpt, training=True)



def train(s:State, iid:int=0)->None:
  """ Train the model by using """
  o = build_outpaths(s)[iid]
  l = mklens(s, build_output_idx=iid)

  max_seq_length = l.max_seq_length.val
  train_data_size = sum(1 for _ in TFRecordDataset(l.datasets.train.syspath))
  train_steps_per_epoch = int(train_data_size // l.train_batch_size.val)


  def _map(record):
    example = tf.io.parse_single_example(record, {
      'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'label_ids': tf.io.FixedLenFeature([], tf.int64)})
    return ({'Input-Token': tf.cast(example['input_ids'],tf.int32),
             'Input-Segment': tf.cast(example['segment_ids'],tf.int32)},
            tf.cast(example['label_ids'], tf.int32))

  dataset_train = \
    TFRecordDataset(l.datasets.train.syspath) \
      .map(_map) \
      .shuffle(100) \
      .repeat() \
      .batch(l.train_batch_size.val, drop_remainder=True) \
      .prefetch(1024)

  dataset_valid = \
    TFRecordDataset(l.datasets.valid.syspath) \
      .map(_map) \
      .shuffle(100) \
      .batch(l.train_batch_size.val, drop_remainder=False)


  s.model_cls.compile(RAdam(lr=mklens(s).lr.val),
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

  tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
                                     write_graph=False, update_freq='batch')

  print('Training')
  h = s.model_cls.fit(
    dataset_train,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=dataset_valid,
    epochs=l.train_epoches.val,
    callbacks=[tensorboard_callback],
    verbose=1)

  protocol_add_hist(l.protocol.syspath, 'train', modelhash(s.model_cls), h)


def evaluate(s:State, iid:int=0)->None:
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

