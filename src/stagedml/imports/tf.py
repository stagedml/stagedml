
from tensorflow import ( Tensor, random_normal_initializer )

from tensorflow.keras.backend import ( clear_session, image_data_format )
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, History
from tensorflow.keras import ( Model, Input )
from tensorflow.keras.models import ( Sequential )
from tensorflow.keras.layers import ( Layer, Conv2D, MaxPool2D, Dropout, Flatten, Dense )
from tensorflow.keras.utils import ( to_categorical )
from tensorflow.summary import ( SummaryWriter, create_file_writer )
from tensorflow.keras.layers import ( Layer, LayerNormalization, Dense )
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import ( TruncatedNormal )
from tensorflow import ( io, data )

Dataset=data.Dataset
FixedLenFeature=io.FixedLenFeature
parse_single_example=io.parse_single_example

def get_single_element(x):
  import tensorflow as tf
  return tf.data.experimental.get_single_element(x)
