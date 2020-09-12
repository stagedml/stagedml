import tensorflow as tf
assert tf.version.VERSION.startswith('2.1') or \
       tf.version.VERSION.startswith('2.2')

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
                                 TensorBoard, Dense, SparseCategoricalAccuracy)

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
  model_test:Model
  optimizer:Any
  config:dict

def build(s:State, iid:int=0):
  """ Build the model.

  FIXME: zero dropout rate if inference """
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
  output_logits = Dense(
    mklens(s).num_classes.val)(
      model_pretrained.get_layer('NSP-Dense').output)
  model_cls = Model(inputs, output_logits)
  # model_cls.summary()

  # FIXME: don't set zero dropout rate for test model
  output_probs = tf.keras.layers.Activation('softmax')(output_logits)
  model_test = Model(inputs, output_probs)
  model_test.summary()

  l = mklens(s, build_output_idx=iid)
  train_data_size = sum(1 for _ in TFRecordDataset(l.datasets.train.syspath))
  train_steps_per_epoch = int(train_data_size // l.train_batch_size.val)
  train_epoches = l.train_epoches.val

  s.config = readjson(mklens(s).bert_config.syspath)
  s.model_pretrained = model_pretrained
  s.model_cls = model_cls
  s.model_test = model_test
  s.optimizer = RAdam(lr=mklens(s).lr.val,
                      total_steps=train_epoches * train_steps_per_epoch)


def cpload(s:State, iid:int=0)->None:
  """ Load GoogleResearch checkpoint into model
  TODO: Explicitly initialize not-affected classification layers
  """
  l = mklens(s, build_output_idx=iid)
  ckpt = l.bert_ckpt.syspath
  load_model_weights_from_checkpoint(
    s.model_pretrained, s.config, ckpt, training=True)


def restore(s:State, rref:RRef)->None:
  """ Load StagedML checkpoint into model """
  s.model_cls.load_weights(mklens(rref).out_ckpt.syspath)


def save(s:State)->None:
  s.model_cls.save(mklens(s).out_savedmodel.syspath,
                   overwrite=True,
                   include_optimizer=False,
                   save_format='tf')

def get_loss_fn(num_classes):
  def classification_loss_fn(labels, logits):
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    return tf.reduce_mean(per_example_loss)
  return classification_loss_fn


def _map(record, max_seq_length):
  example = tf.io.parse_single_example(record, {
    'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
    'segment_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
    'label_ids': tf.io.FixedLenFeature([], tf.int64)})
  return ({'Input-Token': tf.cast(example['input_ids'],tf.int32),
           'Input-Segment': tf.cast(example['segment_ids'],tf.int32)},
          tf.cast(example['label_ids'], tf.int32))

def train(s:State, iid:int=0)->None:
  """ Train the model by using """
  o = build_outpaths(s)[iid]
  l = mklens(s, build_output_idx=iid)

  max_seq_length = l.max_seq_length.val
  train_data_size = sum(1 for _ in TFRecordDataset(l.datasets.train.syspath))
  train_steps_per_epoch = int(train_data_size // l.train_batch_size.val)
  train_epoches = l.train_epoches.val

  dataset_train = \
    TFRecordDataset(l.datasets.train.syspath) \
      .map(partial(_map, max_seq_length=max_seq_length)) \
      .shuffle(100) \
      .repeat() \
      .batch(l.train_batch_size.val, drop_remainder=True) \
      .prefetch(1024)

  dataset_valid = \
    TFRecordDataset(l.datasets.valid.syspath) \
      .map(partial(_map, max_seq_length=max_seq_length)) \
      .shuffle(100) \
      .batch(l.valid_batch_size.val, drop_remainder=False)

  loss_fn = get_loss_fn(num_classes=mklens(s).num_classes.val)

  metric_fn = SparseCategoricalAccuracy('accuracy', dtype=tf.float32)
  s.model_cls.compile(s.optimizer,
                      loss=loss_fn,
                      metrics=metric_fn)

  tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
                                     write_graph=False, update_freq='batch')

  print('Training')
  h = s.model_cls.fit(
    dataset_train,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=dataset_valid,
    epochs=train_epoches,
    callbacks=[tensorboard_callback],
    verbose=1)

  s.model_cls.save_weights(l.out_ckpt.syspath)

  protocol_add_hist(l.out_protocol.syspath, 'train', modelhash(s.model_cls), h)


def test(s:State, iid:int=0)->None:
  """ Evaluate the model """
  c = build_cattrs(s)
  o = build_outpaths(s)[iid]
  l = mklens(s,build_output_idx=iid)

  metrics = [ SparseCategoricalAccuracy('test_accuracy', dtype=tf.float32),
              SparseF1Score(num_classes=c.num_classes, average='micro') ]
  loss_fn = get_loss_fn(num_classes=int(c.num_classes))

  s.model_test.compile(s.optimizer, loss=loss_fn, metrics=metrics)

  dataset_test = \
    TFRecordDataset(l.datasets.test.syspath) \
      .map(partial(_map, max_seq_length=c.max_seq_length)) \
      .shuffle(100) \
      .batch(l.test_batch_size.val, drop_remainder=False)

  print('Testing')
  h = s.model_test.evaluate(dataset_test)

  filewriter = tf.summary.create_file_writer(join(o,'test'))
  with filewriter.as_default():
    for mname,v in zip(s.model_test.metrics_names, h):
      tf.summary.scalar(mname, v, step=0)

  protocol_add_eval(l.out_protocol.syspath, 'test',
                    modelhash(s.model_cls), s.model_test.metrics_names, h)


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
           mklens(tfrecs).bert_vocab.refpath,\
      "Model dictionary path doesn't match the dataset dictionary path"
    num_classes = mklens(tfrecs).num_classes.val
    max_seq_length = mklens(tfrecs).max_seq_length.val
    lr = 2e-5
    train_batch_size = 8
    valid_batch_size = train_batch_size
    test_batch_size = 32
    train_epoches = 3

    out_ckpt = [claim, f'{name}.ckpt']
    out_savedmodel = [claim, f'{name}_savedmodel']
    out_protocol = [promise, 'protocol.json']
    changes = ['+logit-fix', '+save-weights', 'fix-protocol']
    return locals()

  def _make(b:Model)->None:
    build_setoutpaths(b,num_instances)
    for i in range(num_instances):
      build(b,i)
      cpload(b,i)
      train(b,i)
      test(b,i)

  return BertGlue(mkdrv(m,
    config=mkconfig(_config()),
    matcher=protocol_match('test', 'test_accuracy'),
    realizer=build_wrapper_(_make, State)))
