from pylightnix import ( Build, Manager, Lens, DRef, RRef, Build, RefPath, mklens,
    mkdrv, build_wrapper, build_path, mkconfig, match_only, promise,
    build_outpath, realize, instantiate, repl_realize, build_wrapper_,
    repl_buildargs, build_cattrs, match_latest, claim, dircp, build_setoutpaths )

from stagedml.imports import ( walk, abspath, join, Random, partial, cpu_count,
    getpid, makedirs, Pool, bz2_open, json_loads, json_load )
from stagedml.imports.tf import ( Dataset, FixedLenFeature,
    parse_single_example, Input, TruncatedNormal, TensorBoard )
from stagedml.utils import ( concat, batch, flines, dpurge, modelhash, runtb )
from stagedml.core import ( protocol_add, protocol_add_hist,
    protocol_add_eval, protocol_match )
from stagedml.types import ( List, Optional, Wikitext, WikiTFR, Any,
    BertPretrain, Tuple )

from official.nlp.bert.tokenization import FullTokenizer, convert_to_unicode
from official.nlp.optimization import create_optimizer
from official.nlp.bert.create_pretraining_data import (
    create_instances_from_document, write_instance_to_example_files, TrainingInstance )
from official.nlp.modeling.networks.bert_pretrainer import ( BertPretrainer )
from official.nlp.bert.bert_models import ( BertPretrainLossAndMetricLayer )
from official.modeling.model_training_utils import ( run_customized_training_loop )
from absl import logging

from stagedml.models.bert import ( BertLayer, BertInput, BertOutput )

import tensorflow as tf

TokSentence=List[str]
TokDoc=List[TokSentence]

def tokenize_file(input_file:str, tokenizer:FullTokenizer)->List[TokDoc]:
  chars=[chr(ord('a')+x) for x in range(ord('z')-ord('a'))]
  print(chars[getpid()%len(chars)], end='', flush=True)
  all_documents:List[TokDoc] = [[]]
  sentences_skipped=0
  sentences_tokenized=0
  with bz2_open(input_file, "rt", encoding='utf-8') as reader:
    for line in reader:
      d=json_loads(line)
      sentences=d['text'].split('\n')

      for i,s in enumerate(sentences):
        s=s.strip()
        tokens:Optional[TokSentence] = tokenizer.tokenize(s)
        if tokens:
          all_documents[-1].append(tokens)
          sentences_tokenized+=1
        else:
          if len(s)>0:
            sentences_skipped+=1
          else:
            sentences_tokenized+=1
      all_documents.append([])
  all_documents = [x for x in all_documents if len(x)>0]
  return all_documents



def realize_pretraining(b:Build)->None:
  tokenizer = FullTokenizer(vocab_file=mklens(b).vocab_file.syspath,
                            do_lower_case=mklens(b).do_lower_case.val)

  input_files = []
  for root, dirs, filenames in walk(mklens(b).input_folder.syspath, topdown=True):
    for filename in sorted(filenames):
      if filename.endswith('bz2'):
        input_files.append(abspath(join(root, filename)))

  max_seq_length=mklens(b).max_seq_length.val
  dupe_factor=mklens(b).dupe_factor.val
  short_seq_prob=mklens(b).short_seq_prob.val
  masked_lm_prob=mklens(b).masked_lm_prob.val
  max_predictions_per_seq=mklens(b).max_predictions_per_seq.val
  do_whole_word_mask=mklens(b).do_whole_word_mask.val
  rng=Random(mklens(b).random_seed.val)
  ncpu=round((cpu_count() or 1)*0.75)
  outdir=mklens(b).output.syspath
  makedirs(outdir)

  for ibatch,ifiles in enumerate(batch(input_files,mklens(b).input_files_chunk.val)):
    print(f"Reading next {len(ifiles)} files")
    ofiles=[join(outdir,'output-%02d-%02d.tfrecord'%(ibatch,nout)) for nout in \
            range(mklens(b).output_files_per_chunk.val)]

    with Pool(processes=ncpu) as p:
      all_documents=\
          concat(p.map(
            partial(tokenize_file,tokenizer=tokenizer),
            ifiles, chunksize=max(1,len(ifiles)//(4*ncpu))))

    print(f'Read {len(all_documents)} documents')
    print('Shuffling documents')
    rng.shuffle(all_documents)

    print('Producing instances')
    vocab_words = list(tokenizer.vocab.keys())
    instances:List[TrainingInstance] = []
    for _ in range(dupe_factor):
      for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents, document_index, max_seq_length, short_seq_prob,
                masked_lm_prob, max_predictions_per_seq, vocab_words, rng,
                do_whole_word_mask))

    print(f'Produced {len(instances)} instances')
    print('Shuffling instances')
    rng.shuffle(instances)

    print('Writing tfrecords')
    write_instance_to_example_files(
      instances, tokenizer,
      mklens(b).max_seq_length.val,
      mklens(b).max_predictions_per_seq.val,
      ofiles,
      gzip_compress=True)
    print('')


def bert_pretraining_tfrecords(m:Manager, vocab_file:RefPath, wiki:Wikitext)->WikiTFR:

  def _config():
    name='bert_pretraining'
    max_seq_length=128
    max_predictions_per_seq=20
    random_seed=17
    dupe_factor=10
    short_seq_prob=0.1
    masked_lm_prob=0.15
    do_lower_case=True
    do_whole_word_mask=False
    nonlocal vocab_file
    input_folder=mklens(wiki).output.refpath
    input_files_chunk=100
    output_files_per_chunk=4
    output=[promise,'output']
    return mkconfig(locals())

  return WikiTFR(mkdrv(m,
    config=_config(),
    matcher=match_only(),
    realizer=build_wrapper(realize_pretraining)))



def bert_pretraining_dataset(
    tfr:RRef,
    batch_size:int,
    is_training:bool=True,
    input_pipeline_context=None)->Dataset:
  """Creates input dataset from (tf)records files for pretraining."""

  input_pattern=f"{mklens(tfr).output.syspath}/*.tfrecord"
  seq_length=mklens(tfr).max_seq_length.val
  max_predictions_per_seq=mklens(tfr).max_predictions_per_seq.val

  # input_patterns=[]
  # for root, dirs, filenames in walk(mklens(tfr).output.syspath, topdown=True):
  #   for filename in sorted(filenames):
  #     if filename.endswith('tfrecord'):
  #       input_patterns.append(abspath(join(root, filename)))

  # assert len(input_patterns)>0

  dataset = Dataset.list_files(input_pattern, shuffle=is_training)

  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)

  dataset = dataset.repeat()

  # We set shuffle buffer to exactly match total number of
  # training files to ensure that training data is well shuffled.
  input_files:List[str]=tf.io.gfile.glob(input_pattern)
  dataset = dataset.shuffle(len(input_files))

  # In parallel, create tf record dataset for each train files.
  # cycle_length = 8 means that up to 8 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      partial(tf.data.TFRecordDataset, compression_type='GZIP'),
      cycle_length=8, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _decode_record(record):
    """ Decodes a record to a TensorFlow example. """
    example = tf.io.parse_single_example(record, {
      'input_ids': FixedLenFeature([seq_length], tf.int64),
      'input_mask': FixedLenFeature([seq_length], tf.int64),
      'segment_ids': FixedLenFeature([seq_length], tf.int64),
      'masked_lm_positions': FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_ids': FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_weights': FixedLenFeature([max_predictions_per_seq], tf.float32),
      'next_sentence_labels': FixedLenFeature([1], tf.int64),
    })

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t
    return example

  dataset = dataset.map(
      _decode_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _select_data_from_record(record):
    """Filter out features to use for pretraining."""
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids'],
        'masked_lm_positions': record['masked_lm_positions'],
        'masked_lm_ids': record['masked_lm_ids'],
        'masked_lm_weights': record['masked_lm_weights'],
        'next_sentence_labels': record['next_sentence_labels'],
    }
    y = record['masked_lm_weights']
    return (x, y)

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training:
    dataset = dataset.shuffle(100)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(1024)
  return dataset


# __  __           _      _
# |  \/  | ___   __| | ___| |
# | |\/| |/ _ \ / _` |/ _ \ |
# | |  | | (_) | (_| |  __/ |
# |_|  |_|\___/ \__,_|\___|_|




class Model(Build):
  model:tf.keras.Model
  submodel:tf.keras.Model
  strategy:Any
  optimizer:Any
  epoch:int

  def __init__(self, ba):
    super().__init__(ba)
    self.epoch=0


def build(m:Model)->None:
  tf.keras.backend.clear_session()
  c=build_cattrs(m)
  m.strategy=tf.distribute.MirroredStrategy()
  with m.strategy.scope():

    bert_config={
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "max_position_embeddings": 512,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "type_vocab_size": 2,
      "vocab_size": flines(mklens(m).vocab_file.syspath)
    }

    bert_inputs=BertInput(
        Input(shape=(c.max_seq_length,), name='input_word_ids', dtype=tf.int32),
        Input(shape=(c.max_seq_length,), name='input_mask', dtype=tf.int32),
        Input(shape=(c.max_seq_length,), name='input_type_ids', dtype=tf.int32))

    bert_layer = BertLayer(config=bert_config, float_type=tf.float32, name='BERT')

    bert_outputs = bert_layer(bert_inputs)

    # bert_model=BertModelPretrain(ins=bert_inputs,
    #                              outs=bert_outputs,
    #                              embedding_weights=bert_layer.embedding_lookup.embeddings)

    bert_model=tf.keras.Model(inputs=bert_inputs,
                              outputs=[bert_outputs.hidden_output[-1], bert_outputs.cls_output])

    # input_word_ids = Input(shape=(c.max_seq_length,), name='input_word_ids', dtype=tf.int32)
    # input_mask     = Input(shape=(c.max_seq_length,), name='input_mask', dtype=tf.int32)
    # input_type_ids = Input(shape=(c.max_seq_length,), name='input_type_ids', dtype=tf.int32)

    # input_word_ids = Input( shape=(c.max_seq_length,), name='input_word_ids', dtype=tf.int32)
    # input_mask = Input( shape=(c.max_seq_length,), name='input_mask', dtype=tf.int32)
    # input_type_ids = Input( shape=(c.max_seq_length,), name='input_type_ids', dtype=tf.int32)

    masked_lm_positions = Input( shape=(c.max_predictions_per_seq,), name='masked_lm_positions', dtype=tf.int32)
    masked_lm_ids = Input( shape=(c.max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)
    masked_lm_weights = Input( shape=(c.max_predictions_per_seq,), name='masked_lm_weights', dtype=tf.int32)
    next_sentence_labels = Input( shape=(1,), name='next_sentence_labels', dtype=tf.int32)

    bert_pretrain = BertPretrainer(
        network=bert_model,
        embedding_table=bert_layer.embedding_lookup.embeddings,
        num_classes=2,
        num_token_predictions=c.max_predictions_per_seq,
        initializer=TruncatedNormal(stddev=bert_config['initializer_range']),
        output='predictions')

    lm_output, sentence_output = bert_pretrain([
        bert_inputs.input_word_ids, bert_inputs.input_mask, bert_inputs.input_type_ids, masked_lm_positions ])

    pretrain_loss_layer = BertPretrainLossAndMetricLayer(vocab_size=bert_config['vocab_size'])
    output_loss = pretrain_loss_layer(lm_output, sentence_output,
                                      masked_lm_ids, masked_lm_weights, next_sentence_labels)
    model = tf.keras.Model(
        inputs={
            'input_word_ids': bert_inputs.input_word_ids,
            'input_mask': bert_inputs.input_mask,
            'input_type_ids': bert_inputs.input_type_ids,
            'masked_lm_positions': masked_lm_positions,
            'masked_lm_ids': masked_lm_ids,
            'masked_lm_weights': masked_lm_weights,
            'next_sentence_labels': next_sentence_labels,
        },
        outputs=output_loss)

    optimizer = create_optimizer(
        init_lr=c.lr, num_train_steps=c.train_steps_per_epoch*c.train_epoches,
        num_warmup_steps=c.train_warmup_steps)

    model.add_loss(tf.reduce_mean(output_loss))
    model.compile(optimizer)

    m.model=model
    m.submodel=bert_model
    m.optimizer = optimizer

def ftrain(m:Model, init:Optional[RRef]=None)->None:
  c = build_cattrs(m)
  o = build_outpath(m)

  ds = bert_pretraining_dataset(
          mklens(m).tfrecs.rref, c.train_batch_size, is_training=True)

  with m.strategy.scope():
    if init is not None:
      m.epoch = mklens(init).train_epoches.val
      m.model.load_weights(mklens(init).checkpoint_full.syspath)
      dircp(mklens(init).logs.syspath, mklens(m).logs.syspath, make_rw=True)

    tensorboard_callback = TensorBoard(log_dir=mklens(m).logs.syspath,
                                       profile_batch=0,
                                       write_graph=False, update_freq='batch')
    while m.epoch < c.train_epoches:
      print(f"Training {m.epoch+1}/{c.train_epoches}")
      h = m.model.fit(ds,
        initial_epoch=m.epoch,
        epochs=m.epoch+1,
        steps_per_epoch=c.train_steps_per_epoch,
        callbacks=[tensorboard_callback],
        verbose=True)

      m.epoch += 1
      print(f"Saving '{mklens(m).checkpoint_bert.syspath}' after {m.epoch} epoch")
      m.submodel.save_weights(mklens(m).checkpoint_bert.syspath)
      print(f"Saving '{mklens(m).checkpoint_full.syspath}' after {m.epoch} epoch")
      m.model.save_weights(mklens(m).checkpoint_full.syspath)

  protocol_add_hist(mklens(m).protocol.syspath, 'train', modelhash(m.model), h)



def bert_pretrain_wiki_realize(m:Model, init:Optional[RRef]=None)->None:
  build_setoutpaths(m,1); runtb(m)
  build(m); ftrain(m,init) # train(m,init); # evaluate(b); keras_save(b)

def bert_pretrain_wiki(m:Manager, tfrecs:WikiTFR,
    train_epoches:Optional[int]=None,
    resume_rref:Optional[RRef]=None)->BertPretrain:

  train_epoches = 10 if train_epoches is None else train_epoches

  def _config()->dict:
    name = 'bert-pretrain-wiki'
    max_seq_length=mklens(tfrecs).max_seq_length.val
    max_predictions_per_seq=mklens(tfrecs).max_predictions_per_seq.val
    vocab_file = mklens(tfrecs).vocab_file.refpath
    lr = 2e-5
    train_batch_size = 16
    nonlocal train_epoches
    train_steps_per_loop = 200
    train_steps_per_epoch = 1000
    train_warmup_steps = 10000 # FIXME: Sic! why so much?
    protocol = [promise, 'protocol.json']
    checkpoint_full = [claim, 'checkpoint_full.ckpt']
    # checkpoint_opt = [promise, 'checkpoint_opt.pkl']
    checkpoint_bert = [claim, 'checkpoint_bert.ckpt']
    logs = [promise, 'logs']
    version = 6
    return locals()

  return BertPretrain(mkdrv(m,
    config=mkconfig(_config()),
    matcher=match_latest(), # FIXME! protocol_match('evaluate', 'eval_accuracy'),
    realizer=build_wrapper_(partial(bert_pretrain_wiki_realize, init=resume_rref), Model)))





# def train(m:Model)->None:
#   c = build_cattrs(m)
#   o = build_outpath(m)
#   def _model():
#     return m.model, m.submodel
#   def _loss(unused_labels, model_outs, **unused_args):
#     loss_factor=1.0
#     return tf.reduce_mean(model_outs) * loss_factor
#   def _input(ctx:Any)->Dataset:
#     global_batch_size=c.train_batch_size
#     batch_size = ctx.get_per_replica_batch_size(global_batch_size) if ctx else global_batch_size
#     return bert_pretraining_dataset(
#         mklens(m).tfrecs.rref, batch_size, is_training=True)
#   tensorboard_callback = TensorBoard(log_dir=o, profile_batch=0,
#                                      write_graph=False, update_freq='epoch')
#   m.model.optimizer = m.optimizer
#   logging.set_verbosity(logging.INFO)
#   run_customized_training_loop(
#       strategy=m.strategy,
#       model_fn=_model,
#       loss_fn=_loss,
#       model_dir=o,
#       train_input_fn=_input,
#       steps_per_epoch=c.train_steps/c.train_epoches,
#       steps_per_loop=c.train_steps_per_loop,
#       epochs=c.train_epoches,
#       init_checkpoint=None,
#       checkpoint_name_template=None)
#   dpurge(o,'ctl_step.*ckpt', debug=True)
#   with open(join(o,'summaries/training_summary.txt'), 'r') as f:
#     s=json_load(f)
#   protocol_add(mklens(m).protocol.syspath, 'train', modelhash(m.model), result=s)

