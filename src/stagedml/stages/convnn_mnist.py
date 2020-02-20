
import tensorflow as tf
assert tf.version.VERSION.startswith('2.1')

from pylightnix import ( Matcher, Build, Path, RefPath, Config, Manager, RRef,
    DRef, Context, store_cattrs, build_path, build_outpath, build_cattrs, mkdrv,
    rref2path, json_load, build_config, mkconfig, mkbuild, match_only,
    match_best, build_wrapper, tryread, fetchurl )

from stagedml.imports import ( join, clear_session, set_session_config,
    TensorBoard, ModelCheckpoint, to_categorical, np_load, Conv2D, MaxPool2D,
    Dropout, Sequential, Flatten, Dense )

from stagedml.utils.tf import ( runtb, runtensorboard, thash, KerasBuild,
    protocol_add, protocol_add_hist, protocol_add_eval, match_metric, dpurge,
    keras_save, keras_wrapper )

from stagedml.types import ( Mnist, Optional, Any, List, Tuple, Union )


def fetchmnist(m:Manager)->Mnist:
  return Mnist(
    fetchurl(m, name='mnist',
                mode='as-is',
                url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
                sha256='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'))



def mnist_config(mnist:Mnist)->Config:
  dataset:RefPath = [mnist, 'mnist.npz']
  learning_rate = 1e-3
  num_epoches = 6
  return mkconfig(locals())

def mnist_match()->Matcher:
  return match_best('accuracy.txt')

def mnist_realize(b:Build)->None:
  o = build_outpath(b)
  c = build_cattrs(b)

  with np_load(build_path(b, c.dataset), allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
  y_train = to_categorical(y_train, 10)

  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
  y_test = to_categorical(y_test, 10)


  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape = (28,28,1)))
  model.add(Conv2D(64, (3, 3), activation = 'relu'))
  model.add(MaxPool2D(pool_size = (2,2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation = 'relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation = 'softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
  model.fit(x_train, y_train, batch_size = 32, epochs = c.num_epoches, verbose = 0)
  accuracy = model.evaluate(x_test, y_test, verbose = 0)[-1]
  model.save_weights(join(o, 'weights.h5'), save_format='h5')
  with open(join(o,'accuracy.txt'),'w') as f:
    f.write(str(accuracy))

def convnn_mnist(m, mnist:Mnist)->DRef:
  return mkdrv(m, mnist_config(mnist), mnist_match(), build_wrapper(mnist_realize))

