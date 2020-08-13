from tensorflow import ( Tensor, random_normal_initializer )

from tensorflow.keras.backend import ( clear_session, image_data_format )
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, History
from tensorflow.keras import ( Model, Input )
from tensorflow.keras.models import ( Sequential )
from tensorflow.keras.layers import ( Layer, Conv2D, MaxPool2D, Dropout,
    Flatten, Dense )
from tensorflow.keras.utils import ( to_categorical )
from tensorflow.summary import ( SummaryWriter, create_file_writer )
from tensorflow.keras.layers import ( Layer, LayerNormalization, Dense )
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import ( TruncatedNormal )
from tensorflow import ( io, data )
from tensorflow.python.training.checkpoint_utils import ( list_variables,
    load_checkpoint )
from tensorflow.python.framework.errors import ( NotFoundError )
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.python.framework.tensor_util import MakeNdarray
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE )
from tensorboard.backend.event_processing.event_accumulator import (
    ScalarEvent, TensorEvent )
from tensorflow import train

from tensorflow.python.ops.math_ops import ( argmax )

Feature=train.Feature
Features=train.Features
Example=train.Example
Int64List=train.Int64List
Dataset=data.Dataset
FixedLenFeature=io.FixedLenFeature
parse_single_example=io.parse_single_example
TFRecordWriter=io.TFRecordWriter
TFRecordDataset=data.TFRecordDataset

INFINITE_CARDINALITY=data.experimental.INFINITE_CARDINALITY
UNKNOWN_CARDINALITY=data.experimental.UNKNOWN_CARDINALITY
cardinality=data.experimental.cardinality


def get_single_element(x):
  import tensorflow as tf
  return tf.data.experimental.get_single_element(x)

# from official.utils.misc.keras_utils import set_session_config
# from official.nlp.bert.tokenization import FullTokenizer
# from official.nlp.data.classifier_data_lib import \
#     file_based_convert_examples_to_features

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE, DEFAULT_SIZE_GUIDANCE,
    ScalarEvent, TensorEvent )
