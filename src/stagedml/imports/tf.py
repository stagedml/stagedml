
from tensorflow.keras.backend import ( clear_session, image_data_format )
from official.utils.misc.keras_utils import set_session_config

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from tensorflow.keras import ( Model )
from tensorflow.keras.models import ( Sequential )
from tensorflow.keras.layers import ( Conv2D, MaxPool2D, Dropout, Flatten, Dense )
from tensorflow.keras.utils import ( to_categorical )

def get_single_element(x):
  import tensorflow as tf
  return tf.data.experimental.get_single_element(x)
