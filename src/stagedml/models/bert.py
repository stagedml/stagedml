import tensorflow as tf

import json
import six
import copy
import math

from stagedml.imports.tf import ( Tensor )
from stagedml.types import ( Any, NamedTuple, NewType, Tuple )

from official.nlp.bert.configs import BertConfig
from official.nlp.bert_modeling import ( EmbeddingLookup,
    Dense3D, Dense2DProjection )
from official.nlp.bert_modeling import ( get_initializer,
    create_attention_mask_from_input_mask )
from official.modeling import tf_utils


class CustomAttention(tf.keras.layers.Layer):
  def __init__(self,
               num_attention_heads=12,
               size_per_head=64,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               **kwargs):
    super(CustomAttention, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.query_dense = self._projection_dense_layer("query")
    self.key_dense = self._projection_dense_layer("key")
    self.value_dense = self._projection_dense_layer("value")
    self.attention_probs_dropout = tf.keras.layers.Dropout(
        rate=self.attention_probs_dropout_prob)
    super(CustomAttention, self).build(unused_input_shapes)

  def reshape_to_matrix(self, input_tensor):
    """Reshape N > 2 rank tensor to rank 2 tensor for performance."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
      raise ValueError("Input tensor must have at least rank 2."
                       "Shape = %s" % (input_tensor.shape))
    if ndims == 2:
      return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

  def __call__(self, from_tensor, to_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([from_tensor, to_tensor, attention_mask])
    return super(CustomAttention, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer.
    FIXME: Disable the dropouts in non-training mode
    """
    (from_tensor, to_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self.query_dense(from_tensor)

    # `key_tensor` = [B, T, N, H]
    key_tensor = self.key_dense(to_tensor)

    # `value_tensor` = [B, T, N, H]
    value_tensor = self.value_dense(to_tensor)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self.size_per_head)))

    if attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.attention_probs_dropout(attention_probs)

    # `context_layer` = [B, F, N, H]
    context_tensor = tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_tensor)

    return context_tensor, attention_scores

  def _projection_dense_layer(self, name):
    """A helper to define a projection layer."""
    dense3d_impl = Dense3D
    return dense3d_impl(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.size_per_head,
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=False,
        backward_compatible=self.backward_compatible,
        name=name)


class CustomTransformerBlock(tf.keras.layers.Layer):
  """Single transformer layer.

  It has two sub-layers. The first is a multi-head self-attention mechanism, and
  the second is a positionwise fully connected feed-forward network.
  """

  def __init__(self,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(CustomTransformerBlock, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (self.hidden_size, self.num_attention_heads))
    self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""

    dense3d_impl = Dense3D
    dense2dprojection_impl = Dense2DProjection

    self.attention_layer = CustomAttention(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.attention_head_size,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_range=self.initializer_range,
        backward_compatible=self.backward_compatible,
        name="self")

    self.attention_output_dense = dense3d_impl(
        num_attention_heads=self.num_attention_heads,
        size_per_head=int(self.hidden_size / self.num_attention_heads),
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=True,
        backward_compatible=self.backward_compatible,
        name="dense")
    self.attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="LayerNorm", axis=-1, epsilon=1e-12,
            # We do layer norm in float32 for numeric stability.
            dtype=tf.float32))

    self.intermediate_dense = dense2dprojection_impl(
        output_size=self.intermediate_size,
        kernel_initializer=get_initializer(self.initializer_range),
        activation=self.intermediate_activation,
        # Uses float32 so that gelu activation is done in float32.
        fp32_activation=True, name='dense')
    self.output_dense = dense2dprojection_impl(
        output_size=self.hidden_size,
        kernel_initializer=get_initializer(self.initializer_range),
        name='dense')
    self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(axis=-1,
        epsilon=1e-12, dtype=tf.float32, name='LayerNorm')
    super(CustomTransformerBlock, self).build(unused_input_shapes)

  def common_layers(self):
    """Explicitly gets all layer objects inside a Transformer encoder block."""
    return [
        self.attention_layer, self.attention_output_dense,
        self.attention_dropout, self.attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_dropout,
        self.output_layer_norm
    ]

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(CustomTransformerBlock, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    (input_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)
    with tf.name_scope('attention'):
      attention_output, attention_scores = self.attention_layer(
          from_tensor=input_tensor,
          to_tensor=input_tensor,
          attention_mask=attention_mask)
      with tf.name_scope('output'):
        attention_output = self.attention_output_dense(attention_output)
        attention_output = self.attention_dropout(attention_output)
        # Use float32 in keras layer norm and the gelu activation in the
        # intermediate dense layer for numeric stability
        attention_output = self.attention_layer_norm(input_tensor +
                                                     attention_output)
        if self.float_type == tf.float16:
          attention_output = tf.cast(attention_output, tf.float16)

    with tf.name_scope('intermediate'):
      intermediate_output = self.intermediate_dense(attention_output)
      if self.float_type == tf.float16:
        intermediate_output = tf.cast(intermediate_output, tf.float16)

    with tf.name_scope('output'):
      layer_output = self.output_dense(intermediate_output)
      layer_output = self.output_dropout(layer_output)
      # Use float32 in keras layer norm for numeric stability
      layer_output = self.output_layer_norm(layer_output + attention_output)
      if self.float_type == tf.float16:
        layer_output = tf.cast(layer_output, tf.float16)
    return layer_output, attention_scores


class CustomTransformer(tf.keras.layers.Layer):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  """

  def __init__(self,
               num_hidden_layers=12,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(CustomTransformer, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.layers = []
    for i in range(self.num_hidden_layers):
      self.layers.append(
          CustomTransformerBlock(
              hidden_size=self.hidden_size,
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self.intermediate_size,
              intermediate_activation=self.intermediate_activation,
              hidden_dropout_prob=self.hidden_dropout_prob,
              attention_probs_dropout_prob=self.attention_probs_dropout_prob,
              initializer_range=self.initializer_range,
              backward_compatible=self.backward_compatible,
              float_type=self.float_type,
              name=("layer_%d" % i)))
    super(CustomTransformer, self).build(unused_input_shapes)

  def __call__(self, input_tensor:Tensor, attention_mask:bool=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(CustomTransformer, self).__call__(inputs=inputs, **kwargs)

  def call(self, inputs)->Any:
    """Implements call() for the layer.

    Args:
      inputs: packed inputs.
    Returns:
      Output tensor of the last layer or a list of output tensors.
    """
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_tensor = unpacked_inputs[0]
    attention_mask = unpacked_inputs[1]
    output_tensor = input_tensor

    all_layer_outputs = []
    for layer in self.layers:
      output_tensor, attention_scores = layer(output_tensor, attention_mask)
      all_layer_outputs.append((output_tensor, attention_scores))

    return all_layer_outputs

BertInput=NamedTuple('BertInput',[('input_word_ids',Tensor),
                                  ('input_mask',Tensor),
                                  ('input_type_ids',Tensor)])

BertOutput=NamedTuple('BertOutput',[('cls_output',Tensor),
                                    ('embedding_output',Tensor),
                                    ('attention_output',Tensor),
                                    ('hidden_output',Tensor)])

class CustomEmbeddingPostprocessor(tf.keras.layers.Layer):
  """Performs various post-processing on a word embedding tensor."""

  def __init__(self,
               word_embedding_width,
               use_type_embeddings=False,
               token_type_vocab_size=None,
               use_position_embeddings=True,
               max_position_embeddings=512,
               dropout_prob=0.0,
               initializer_range=0.02,
               **kwargs):
    super().__init__(**kwargs)

    self.use_type_embeddings = use_type_embeddings
    self.token_type_vocab_size = token_type_vocab_size
    self.use_position_embeddings = use_position_embeddings
    self.max_position_embeddings = max_position_embeddings
    self.dropout_prob = dropout_prob
    self.initializer_range = initializer_range
    self.initializer = get_initializer(self.initializer_range)
    self.word_embedding_width = word_embedding_width
    self.type_embeddings = None
    self.position_embeddings = None

    if self.use_type_embeddings and not self.token_type_vocab_size:
      raise ValueError("If `use_type_embeddings` is True, then "
                       "`token_type_vocab_size` must be specified.")

  def build(self,unused):
    """Implements build() for the layer."""
    # (word_embeddings_shape, _) = input_shapes
    # width = word_embeddings_shape.as_list()[-1]
    if self.use_type_embeddings:
      self.type_embeddings = self.add_weight(
          "token_type_embeddings",
          shape=[self.token_type_vocab_size, self.word_embedding_width],
          initializer=get_initializer(self.initializer_range),
          dtype=self.dtype)

    self.position_embeddings = None
    if self.use_position_embeddings:
      self.position_embeddings = self.add_weight(
          "position_embeddings",
          shape=[self.max_position_embeddings, self.word_embedding_width],
          initializer=get_initializer(self.initializer_range),
          dtype=self.dtype)

    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="LayerNorm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    self.output_dropout = tf.keras.layers.Dropout(
        rate=self.dropout_prob, dtype=tf.float32)
    super().build(unused)

  def __call__(self, word_embeddings, token_type_ids=None, **kwargs):
    inputs = tf_utils.pack_inputs([word_embeddings, token_type_ids])
    return super().__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    word_embeddings = unpacked_inputs[0]
    token_type_ids = unpacked_inputs[1]
    input_shape = tf_utils.get_shape_list(word_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = word_embeddings
    if self.use_type_embeddings:
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      token_type_embeddings = tf.gather(self.type_embeddings,
                                        flat_token_type_ids)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
      output += token_type_embeddings

    if self.use_position_embeddings:
      position_embeddings = tf.expand_dims(
          tf.slice(self.position_embeddings, [0, 0], [seq_length, width]),
          axis=0)

      output += position_embeddings

    output = self.output_layer_norm(output)
    output = self.output_dropout(output)

    return output

class BertLayer(tf.keras.layers.Layer):
  def __init__(self, config, float_type=tf.float32, **kwargs):
    super(BertLayer, self).__init__(**kwargs)
    self.config = (
        BertConfig.from_dict(config)
        if isinstance(config, dict) else copy.deepcopy(config))
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embedding_lookup = EmbeddingLookup(
        vocab_size=self.config.vocab_size,
        embedding_size=self.config.hidden_size,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="word_embeddings")
    self.embedding_postprocessor = CustomEmbeddingPostprocessor(
        word_embedding_width=self.config.hidden_size,
        use_type_embeddings=True,
        token_type_vocab_size=self.config.type_vocab_size,
        use_position_embeddings=True,
        max_position_embeddings=self.config.max_position_embeddings,
        dropout_prob=self.config.hidden_dropout_prob,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="embeddings")
    self.encoder = CustomTransformer(
        num_hidden_layers=self.config.num_hidden_layers,
        hidden_size=self.config.hidden_size,
        num_attention_heads=self.config.num_attention_heads,
        intermediate_size=self.config.intermediate_size,
        intermediate_activation=self.config.hidden_act,
        hidden_dropout_prob=self.config.hidden_dropout_prob,
        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        initializer_range=self.config.initializer_range,
        backward_compatible=self.config.backward_compatible,
        float_type=self.float_type,
        name="encoder")
    self.pooler_dense = tf.keras.layers.Dense(
        units=self.config.hidden_size,
        activation="tanh",
        kernel_initializer=get_initializer(self.config.initializer_range))
    super(BertLayer, self).build(unused_input_shapes)

  def __call__(self, bert_input:BertInput)->BertOutput:
    return super(BertLayer, self).__call__(bert_input)

  def call(self, inputs):
    # unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_word_ids = inputs.input_word_ids #  unpacked_inputs[0]
    input_mask = inputs.input_mask         # unpacked_inputs[1]
    input_type_ids = inputs.input_type_ids # unpacked_inputs[2]

    word_embeddings = self.embedding_lookup(input_word_ids)
    embedding_tensor = self.embedding_postprocessor(
        word_embeddings=word_embeddings, token_type_ids=input_type_ids)
    if self.float_type == tf.float16:
      embedding_tensor = tf.cast(embedding_tensor, tf.float16)
    attention_mask = None
    if input_mask is not None:
      attention_mask = create_attention_mask_from_input_mask(
          input_word_ids, input_mask)

    sequence_output = self.encoder(embedding_tensor, attention_mask)
    embedding_output = embedding_tensor
    hidden_output, attention_output = list(zip(*sequence_output))

    first_token_tensor = tf.squeeze(hidden_output[-1][:, 0:1, :], axis=1)
    with tf.name_scope('pooler'):
      pooled_output = self.pooler_dense(first_token_tensor)

    return BertOutput(pooled_output, embedding_output, attention_output, hidden_output)

  def get_config(self):
    config = {"config": self.config.to_dict()}
    base_config = super(BertLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def classification_logits(config, num_labels, pooled_input):
  t = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)(pooled_input)
  logits = tf.keras.layers.Dense(num_labels,
                                 kernel_initializer='glorot_uniform',
                                 activation=None,
                                 name='output')(t)
  return logits



class BertModel:
  def __init__(self, ins:BertInput, outs:BertOutput)->None:
    self.model=tf.keras.Model(inputs=ins, outputs=outs)
  def __call__(self, ins:BertInput)->BertOutput:
    return BertOutput(*self.model(ins))


# class BertModelPretrain:
#   def __init__(self, ins:BertInput, outs:BertOutput, embedding_weights:Tensor)->None:
#     def _get_embedding_table(self)->Tensor:
#       return embedding_weights
#     self.model=tf.keras.Model(inputs=ins, outputs=outs)
#     self.model.get_embedding_table = _get_embedding_table
#   def __call__(self, ins:BertInput)->Tuple[Tensor,Tensor]:
#     print('XXXXXXXXXXXXXXXXXXXX')
#     bo=BertOutput(*self.model(ins))
#     return (bo.hidden_output[-1], bo.cls_output)
#     # self.embedding_weights=embedding_weights
#     # self.inputs=self.model.inputs
#     # self.outputs=self.model.outputs

