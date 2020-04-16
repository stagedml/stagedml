import tensorflow as tf
from stagedml.imports import ( Layer, Tensor, partial )

from stagedml.models.transformer.imports import (
    padded_neg_log_perplexity, padded_accuracy, padded_accuracy_top5,
    padded_sequence_accuracy )

class Metrics(Layer):
  """Custom a layer of metrics for Transformer model."""

  def __init__(self, vocab_size)->None:
    """"Builds metric layer."""
    super().__init__()
    self.vocab_size = vocab_size
    neg_log_perplexity = partial(
        padded_neg_log_perplexity, vocab_size=self.vocab_size)
    self.metric_mean_fns:list = [
        (tf.keras.metrics.Mean("accuracy"), padded_accuracy),
        (tf.keras.metrics.Mean("accuracy_top5"), padded_accuracy_top5),
        (tf.keras.metrics.Mean("accuracy_per_sequence"), padded_sequence_accuracy),
        (tf.keras.metrics.Mean("neg_log_perplexity"), neg_log_perplexity),
    ]

  def call(self, logits:Tensor, targets:Tensor)->Tensor:
    for mean, fn in self.metric_mean_fns:
      m = mean(*fn(logits, targets))
      self.add_metric(m)
    return logits

