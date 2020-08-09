
import tensorflow as tf
assert tf.version.VERSION.startswith('2.1') or \
       tf.version.VERSION.startswith('2.2')

from pylightnix import ( Stage, Matcher, Build, Path, RefPath, Config, Manager,
    RRef, DRef, Context, store_cattrs, build_path, build_outpath, build_cattrs,
    mkdrv, rref2path, json_load, build_config, mkconfig, mkbuild, match_only,
    match_best, build_wrapper_, tryread, fetchurl, mklens, promise,
    instantiate, realize, redefine )

from stagedml.imports.sys import ( join, np_load, partial )
from stagedml.imports.tf import ( clear_session,
    TensorBoard, ModelCheckpoint, to_categorical, Conv2D, MaxPool2D,
    Dropout, Sequential, Flatten, Dense )

from stagedml.utils.sys import ( dpurge )
from stagedml.utils.tf import ( runtb, runtensorboard, thash, modelhash )

from stagedml.core import ( protocol_add, protocol_add_hist,
    protocol_add_eval, protocol_match )

from stagedml.types import ( ConvnnMnist, Mnist, Optional, Any, List, Tuple,
    Union )


class Model(Build):
  model:Sequential
  x_train:Any
  y_train:Any
  x_test:Any
  y_test:Any



def mnist_train(b:Model)->None:
  o = build_outpath(b)
  c = build_cattrs(b)

  with np_load(build_path(b, c.dataset), allow_pickle=True) as f:
    b.x_train, b.y_train = f['x_train'], f['y_train']
    b.x_test, b.y_test = f['x_test'], f['y_test']

  b.x_train = b.x_train.reshape(b.x_train.shape[0], 28, 28, 1).astype('float32') / 255
  b.y_train = to_categorical(b.y_train, 10)

  b.x_test = b.x_test.reshape(b.x_test.shape[0], 28, 28, 1).astype('float32') / 255
  b.y_test = to_categorical(b.y_test, 10)


  print('x_train shape:', b.x_train.shape)
  print(b.x_train.shape[0], 'train samples')
  print(b.x_test.shape[0], 'test samples')

  model = Sequential()
  b.model = model
  model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape = (28,28,1)))
  model.add(Conv2D(64, (3, 3), activation = 'relu'))
  model.add(MaxPool2D(pool_size = (2,2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation = 'relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation = 'softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  callbacks = [
    TensorBoard(log_dir=o),
    ModelCheckpoint(
      monitor='val_accuracy',
      filepath=join(o, "checkpoint.ckpt"),
      save_weights_only=True,
      save_best_only=True,
      verbose=True)]
  h=model.fit(b.x_train, b.y_train,
      batch_size=32,
      epochs=c.num_epoches,
      verbose=True,
      callbacks=callbacks,
      validation_split=0.2)
  protocol_add_hist(mklens(b).protocol.syspath, 'train', modelhash(model), h)

def mnist_eval(b:Model):
  o = build_outpath(b)
  b.model.load_weights(join(o, "checkpoint.ckpt"))
  metrics = b.model.evaluate(b.x_test, b.y_test, verbose = 0)
  protocol_add_eval(mklens(b).protocol.syspath, 'eval', modelhash(b.model),
                    metric_names=b.model.metrics_names, result=metrics)

def mnist_realize(b:Model):
  mnist_train(b)
  mnist_eval(b)

def convnn_mnist(m:Manager, mnist:Mnist)->ConvnnMnist:

  def _config()->dict:
    nonlocal mnist
    name = 'convnn-'+mklens(mnist).name.val
    dataset:RefPath = [mnist, 'mnist.npz']
    learning_rate = 1e-3
    num_epoches = 6
    version = 6
    protocol = [promise, 'protocol.json']
    return locals()

  return ConvnnMnist(mkdrv(m, mkconfig(_config()), protocol_match('eval','accuracy'),
    build_wrapper_(mnist_realize,Model)))



