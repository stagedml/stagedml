import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from tensorflow.keras.layers import Layer

class BertSquadLogitsLayer(Layer):
  """Returns a layer that computes custom logits for BERT squad model."""

  def __init__(self, initializer=None, float_type=tf.float32, **kwargs):
    super(BertSquadLogitsLayer, self).__init__(**kwargs)
    self.initializer = initializer
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.final_dense = tf.keras.layers.Dense(
        units=2, kernel_initializer=self.initializer, name='final_dense')
    super(BertSquadLogitsLayer, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""
    sequence_output = inputs

    input_shape = sequence_output.shape.as_list()
    sequence_length = input_shape[1]
    num_hidden_units = input_shape[2]

    final_hidden_input = tf.keras.backend.reshape(sequence_output,
                                                  [-1, num_hidden_units])
    logits = self.final_dense(final_hidden_input)
    logits = tf.keras.backend.reshape(logits, [-1, sequence_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])
    unstacked_logits = tf.unstack(logits, axis=0)
    if self.float_type == tf.float16:
      unstacked_logits = tf.cast(unstacked_logits, tf.float32)
    return unstacked_logits[0], unstacked_logits[1]

