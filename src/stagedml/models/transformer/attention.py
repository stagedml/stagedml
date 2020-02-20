import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from stagedml.models.transformer.imports import ( Tensor, Model, Dense3D )


class Attention:
  """Multi-headed attention layer."""

  def __init__(self, path, hidden_size, num_heads, attention_dropout):
    """Initialize Attention.

    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout

    size_per_head = self.hidden_size // self.num_heads
    self.query_dense_layer = Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name=path+'/'+"query")
    self.key_dense_layer = Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name=path+'/'+"key")
    self.value_dense_layer = Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name=path+'/'+"value")
    self.output_dense_layer = Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, output_projection=True, name=path+'/'+"output_transform")

  def __call__(self, query_input:Tensor, source_input:Tensor, bias, training, cache=None,
               decode_loop_step=None)->Tensor:
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].
    query = self.query_dense_layer(query_input)
    key = self.key_dense_layer(source_input)
    value = self.value_dense_layer(source_input)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      if decode_loop_step is not None:
        cache_k_shape = cache["k"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
            [1, cache_k_shape[1], 1, 1])
        key = cache["k"] + key * indices
        cache_v_shape = cache["v"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
            [1, cache_v_shape[1], 1, 1])
        value = cache["v"] + value * indices
      else:
        key = tf.concat([tf.cast(cache["k"], key.dtype), key], axis=1)
        value = tf.concat([tf.cast(cache["v"], value.dtype), value], axis=1)

      # Update cache
      cache["k"] = key
      cache["v"] = value

    # Scale query to prevent the dot product between query and key from growing
    # too large.
    depth = (self.hidden_size // self.num_heads)
    query *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.einsum("BTNH,BFNH->BNFT", key, query)
    logits += bias
    # Note that softmax internally performs math operations using float32
    # for numeric stability. When training with float16, we keep the input
    # and output in float16 for better performance.
    weights = tf.nn.softmax(logits, name="attention_weights")
    if training:
      weights = tf.nn.dropout(weights, rate=self.attention_dropout)
    attention_output = tf.einsum("BNFT,BTNH->BFNH", weights, value)

    # Run the outputs through another linear projection layer. Recombining heads
    # is automatically done --> [batch_size, length, hidden_size]
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention:
  def __init__(self, **kwargs)->None:
    self.attn = Attention(**kwargs)

  def __call__(self, query_input:Tensor, bias, training, cache=None,
                     decode_loop_step=None)->Tensor:
    return self.attn(query_input, query_input, bias, training, cache, decode_loop_step)

