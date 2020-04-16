import tensorflow as tf
assert tf.version.VERSION.startswith('2.1') or \
       tf.version.VERSION.startswith('2.2')

from .imports import *
from .model import *
from .data import *
