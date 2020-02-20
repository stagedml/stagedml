import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from stagedml.models.transformer.imports import ( Tensor, Dense )

from typing import List

class FeedForwardNetwork:

  def __init__(self, path, hidden_size, filter_size, relu_dropout)->None:
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout

    self.filter_dense_layer = Dense(
        self.filter_size,
        use_bias=True,
        activation=tf.nn.relu,
        name=path+"/filter_layer")
    self.output_dense_layer = Dense(
        self.hidden_size, use_bias=True, name=path+"/output_layer")

  def __call__(self, x:Tensor, training:bool)->Tensor:
    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    output = self.filter_dense_layer(x)
    if training:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    output = self.output_dense_layer(output)

    return output

