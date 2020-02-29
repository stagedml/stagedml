import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from stagedml.models.transformer.imports import ( Model, Layer, Tensor,
    transformer_loss, LayerNormalization, get_padding_bias, get_padding,
    get_position_encoding, get_decoder_self_attention_bias,
    sequence_beam_search, EOS_ID, LearningRateSchedule, Adam, train_input_fn,
    map_data_for_transformer_fn )

from stagedml.models.transformer.metrics import Metrics
from stagedml.models.transformer.attention import Attention, SelfAttention
from stagedml.models.transformer.ffn import FeedForwardNetwork
from stagedml.models.transformer.embedding import EmbeddingSharedWeights

from typing import Any, List, Tuple


def create_train_model(params:dict)->Model:
  """Creates transformer model for training."""
  with tf.name_scope("model"):
    inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
    targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
    internal_model = TransformerLayer(params, name='transformerv2')
    logits = internal_model([inputs, targets], training=True)
    vocab_size = params["vocab_size"]
    label_smoothing = params["label_smoothing"]
    if params["enable_metrics_in_training"]:
      logits = Metrics(vocab_size)(logits, targets)
    logits = tf.keras.layers.Lambda(lambda x: x, name="logits", dtype=tf.float32)(logits)
    model = Model([inputs, targets], logits)
    # TODO(reedwm): Can we do this loss in float16 instead of float32?
    loss = transformer_loss(logits, targets, label_smoothing, vocab_size)
    model.add_loss(loss)
    return model

def create_eval_model(params:dict)->Model:
  with tf.name_scope("model"):
    inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
    internal_model = TransformerLayer(params, name="transformer_v2")
    ret = internal_model([inputs], training=False)
    outputs, scores = ret["outputs"], ret["scores"]
    return tf.keras.Model(inputs, [outputs, scores])


def create_optimizer(params:dict)->Adam:
  """Creates optimizer."""
  # TODO(b/139414679): Explore the difference between using
  # LearningRateSchedule and callback for GPU runs, and try to merge them.
  lr_schedule = LearningRateSchedule(
      params["learning_rate"],
      params["hidden_size"],
      params["learning_rate_warmup_steps"])
  opt = Adam(params["learning_rate"],
             params["optimizer_adam_beta1"],
             params["optimizer_adam_beta2"],
             epsilon=params["optimizer_adam_epsilon"])

  assert not (params["dtype"] == tf.float16)
  return opt

class TransformerLayer(tf.keras.layers.Layer):

  def __init__(self, params, **kwargs)->None:
    super(TransformerLayer, self).__init__(**kwargs)
    self.params = params
    self.embedding_softmax_layer = EmbeddingSharedWeights(
        params["vocab_size"], params["hidden_size"])
    self.encoder_stack = EncoderStack(params)
    self.decoder_stack = DecoderStack(params)

  def call(self, inputs:Tensor, training)->Tensor:
    if len(inputs) == 2:
      inputs, targets = inputs[0], inputs[1]
    else:
      # Decoding path.
      inputs, targets = inputs[0], None
      if self.params["padded_decode"]:
        if not self.params["num_replicas"]:
          raise NotImplementedError(
              "Padded decoding on CPU/GPUs is not supported.")
        decode_batch_size = int(self.params["decode_batch_size"] /
                                self.params["num_replicas"])
        inputs.set_shape([
            decode_batch_size, self.params["decode_max_length"]
        ])

    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    with tf.name_scope("Transformer"):
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      attention_bias = get_padding_bias(inputs)

      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs = self.encode(inputs, attention_bias, training)
      # Generate output sequence if targets is None, or return logits if target
      # sequence is known.
      if targets is None:
        return self.predict(encoder_outputs, attention_bias, training)
      else:
        logits = self.decode(targets, encoder_outputs, attention_bias, training)
        return logits

  def encode(self, inputs, attention_bias, training:bool)->Tensor:
    """Generate continuous representation for inputs.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
      training: boolean, whether in training mode or not.

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      embedded_inputs = self.embedding_softmax_layer(inputs)
      embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])
      inputs_padding = get_padding(inputs)
      attention_bias = tf.cast(attention_bias, self.params["dtype"])

      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = get_position_encoding(length, self.params["hidden_size"])
        pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
        encoder_inputs = embedded_inputs + pos_encoding

      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self.params["layer_postprocess_dropout"])

      # x = LayerNormalization(epsilon=1e-6, dtype="float32")(encoder_inputs)

      return self.encoder_stack(
          encoder_inputs, attention_bias, inputs_padding, training=training)

  def decode(self, targets, encoder_outputs, attention_bias, training:bool)->Tensor:
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence. int tensor with shape
        [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence. float tensor
        with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
      training: boolean, whether in training mode or not.

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope("decode"):
      # Prepare inputs to decoder layers by shifting targets, adding positional
      # encoding and applying dropout.
      decoder_inputs = self.embedding_softmax_layer(targets)
      decoder_inputs = tf.cast(decoder_inputs, self.params["dtype"])
      attention_bias = tf.cast(attention_bias, self.params["dtype"])
      with tf.name_scope("shift_targets"):
        # Shift targets to the right, and remove the last element
        decoder_inputs = tf.pad(decoder_inputs,
                                [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(decoder_inputs)[1]
        pos_encoding = get_position_encoding(
            length, self.params["hidden_size"])
        pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
        decoder_inputs += pos_encoding
      if training:
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, rate=self.params["layer_postprocess_dropout"])

      # Run values
      decoder_self_attention_bias = get_decoder_self_attention_bias(
          length, dtype=self.params["dtype"])
      outputs = self.decoder_stack(
          decoder_inputs,
          encoder_outputs,
          decoder_self_attention_bias,
          attention_bias,
          training=training)
      logits = self.embedding_softmax_layer(outputs, mode="linear")
      logits = tf.cast(logits, tf.float32)
      return logits

  def _get_symbols_to_logits_fn(self, max_decode_length, training)->Any:
    """Returns a decoding function that calculates logits of the next tokens."""

    timing_signal = \
        get_position_encoding(max_decode_length + 1,
                              self.params["hidden_size"])
    timing_signal = tf.cast(timing_signal, self.params["dtype"])
    decoder_self_attention_bias = \
        get_decoder_self_attention_bias(max_decode_length,
                                        dtype=self.params["dtype"])

    # TODO(b/139770046): Refactor code with better naming of i.
    def symbols_to_logits_fn(ids, i, cache)->Tuple[Tensor,Tensor]:
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1].
        i: Loop index.
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_softmax_layer(decoder_input)

      if self.params["padded_decode"]:
        timing_signal_shape = timing_signal.shape.as_list()
        decoder_input += tf.slice(timing_signal, [i, 0],
                                  [1, timing_signal_shape[1]])

        bias_shape = decoder_self_attention_bias.shape.as_list()
        self_attention_bias = tf.slice(
            decoder_self_attention_bias, [0, 0, i, 0],
            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
      else:
        decoder_input += timing_signal[i:i + 1]

        self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_outputs = self.decoder_stack(
          decoder_input,
          cache.get("encoder_outputs"),
          self_attention_bias,
          cache.get("encoder_decoder_attention_bias"),
          training=training,
          cache=cache,
          decode_loop_step=i if self.params["padded_decode"] else None)
      logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias, training)->Tensor:
    """Return predicted sequence."""
    encoder_outputs = tf.cast(encoder_outputs, self.params["dtype"])
    if self.params["padded_decode"]:
      batch_size = encoder_outputs.shape.as_list()[0]
      input_length = encoder_outputs.shape.as_list()[1]
    else:
      batch_size = tf.shape(encoder_outputs)[0]
      input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + self.params["extra_decode_length"]
    encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                             self.params["dtype"])

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(
        max_decode_length, training)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    # pylint: disable=g-complex-comprehension
    init_decode_length = (
        max_decode_length if self.params["padded_decode"] else 0)
    num_heads = self.params["num_heads"]
    dim_per_head = self.params["hidden_size"] // num_heads
    cache = {
        "layer_%d" % layer: {
            "k":
                tf.zeros([
                    batch_size, init_decode_length, num_heads, dim_per_head
                ],
                         dtype=self.params["dtype"]),
            "v":
                tf.zeros([
                    batch_size, init_decode_length, num_heads, dim_per_head
                ],
                         dtype=self.params["dtype"])
        } for layer in range(self.params["num_hidden_layers"])
    }
    # pylint: enable=g-complex-comprehension

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self.params["vocab_size"],
        beam_size=self.params["beam_size"],
        alpha=self.params["alpha"],
        max_decode_length=max_decode_length,
        eos_id=EOS_ID,
        padded_decode=self.params["padded_decode"],
        dtype=self.params["dtype"])

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    top_decoded_ids_padded = tf.pad(top_decoded_ids,
        [[0,0],[0,self.params['max_length']-tf.shape(top_decoded_ids)[1]]])

    return {"outputs": top_decoded_ids_padded, "scores": top_scores}




class EncoderStack(Layer):

  def __init__(self, params:dict)->None:
    super().__init__()
    self.params = params
    self.layers = []
    params = self.params
    for i in range(params["num_hidden_layers"]):
      # Create sublayers for each layer.
      self_attention_layer = SelfAttention(
          hidden_size=params["hidden_size"],
          num_heads=params["num_heads"],
          attention_dropout=params["attention_dropout"])

      feed_forward_network = FeedForwardNetwork(
          hidden_size=params["hidden_size"],
          filter_size=params["filter_size"],
          relu_dropout=params["relu_dropout"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
      ])

    # Create final layer normalization layer.
    self.output_normalization = LayerNormalization(epsilon=1e-6, dtype="float32")


  def call(self, encoder_inputs:Tensor,
                 attention_bias,
                 inputs_padding,
                 training:bool)->Tensor:

    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.name_scope("layer_%d" % n):
        with tf.name_scope("self_attention"):
          encoder_inputs = self_attention_layer(
              encoder_inputs, attention_bias, training=training)
        with tf.name_scope("ffn"):
          encoder_inputs = feed_forward_network(
              encoder_inputs, training=training)

    return self.output_normalization(encoder_inputs)




class DecoderStack(Layer):

  def __init__(self, params:dict)->None:
    super().__init__()
    self.params = params
    self.layers = []

    """Builds the decoder stack."""
    params = self.params
    for i in range(params["num_hidden_layers"]):
      self_attention_layer = SelfAttention(
          hidden_size=params["hidden_size"],
          num_heads=params["num_heads"],
          attention_dropout=params["attention_dropout"])
      enc_dec_attention_layer = Attention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      feed_forward_network = FeedForwardNetwork(
          params["hidden_size"], params["filter_size"], params["relu_dropout"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(enc_dec_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
      ])
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")

  def call(self,
      decoder_inputs:Tensor,
      encoder_outputs:Tensor,
      decoder_self_attention_bias,
      attention_bias,
      training,
      cache=None,
      decode_loop_step=None)->Tensor:

    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.name_scope(layer_name):
        with tf.name_scope("self_attention"):
          decoder_inputs = self_attention_layer(
              decoder_inputs,
              decoder_self_attention_bias,
              training=training,
              cache=layer_cache,
              decode_loop_step=decode_loop_step)
        with tf.name_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs,
              encoder_outputs,
              attention_bias,
              training=training)
        with tf.name_scope("ffn"):
          decoder_inputs = feed_forward_network(
              decoder_inputs, training=training)

    return self.output_normalization(decoder_inputs)


class PrePostProcessingWrapper(Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params)->None:
    super().__init__()
    self.layer = layer
    self.params = params
    self.postprocess_dropout = params["layer_postprocess_dropout"]
    # Create normalization layer
    self.layer_norm = LayerNormalization(epsilon=1e-6, dtype="float32")

  def call(self, x:Tensor, *args, **kwargs)->Tensor:
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    print(type(x), x.dtype)
    print(x.shape)
    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if training:
      y = tf.nn.dropout(y, rate=self.postprocess_dropout)
    return x + y




