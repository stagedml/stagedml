import tensorflow as tf
from stagedml.imports.tf import ( Layer, Tensor, Dense )

from typing import List

class FeedForwardNetwork(Layer):

  def __init__(self, hidden_size, filter_size, relu_dropout)->None:
    super().__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout

    self.filter_dense_layer = Dense(
        self.filter_size,
        use_bias=True,
        activation=tf.nn.relu,
        name="filter_layer")
    self.output_dense_layer = Dense(
        self.hidden_size, use_bias=True, name="output_layer")

  def call(self, x:Tensor, training:bool)->Tensor:
    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    output = self.filter_dense_layer(x)
    if training:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    output = self.output_dense_layer(output)

    return output

