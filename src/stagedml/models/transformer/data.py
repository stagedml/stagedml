import tensorflow as tf

from stagedml.models.transformer.imports import ( LearningRateScheduler,
    LearningRateFn, BASE_PARAMS, Subtokenizer, train_input_fn,
    map_data_for_transformer_fn )

from stagedml.models.transformer.model import ( create_train_model,
    create_eval_model, create_optimizer )



def train_ds(params:dict):
  train_ds = train_input_fn(params)
  train_ds = \
      train_ds.map(
          map_data_for_transformer_fn,
          num_parallel_calls=params["num_parallel_calls"])
  return train_ds


def eval_ds(subtokenizer:Subtokenizer, eval_file:str, batch_size:int, params:dict):
  def _gen():
    with open(eval_file,'r') as f:
      for line in f:
        yield {'inputs':subtokenizer.encode(line, add_eos=True)}

  def _repeat(x):
    return {k:x for k in ['inputs']}

  ds=tf.data.Dataset.from_generator(_gen,
      output_types=_repeat(tf.int64),
      output_shapes=_repeat(tf.TensorShape([None]))
      )

  # def _filter_max_length(example):
  #   return tf.size(example['inputs']) <= params['max_length']
  # ds=ds.filter(_filter_max_length)

  print('Even newer ds')
  ds=ds.padded_batch(batch_size=batch_size,
                     padded_shapes=_repeat([None]),
                     drop_remainder=False)
  return ds
