from os import environ
from os.path import join, islink, isdir, isfile
from stagedml.utils.files import assert_link

import tensorflow as tf
assert tf.version.VERSION.startswith('2.1') or \
       tf.version.VERSION.startswith('2.2'), \
    (f"StagedML requires TensorFlow versions '2.1.*' or '2.2.*', but got"
     f"{tf.version.VERSION}. You may need to install it using `sudo -H pip3"
     f"install tensorflow-gpu`, or build from sources which are located in "
     f"`3rdparty/tensorflow` submodule. For building, please check helper "
     f"shell-functions defined in `env.sh`, namely `conftf`, `buildtf`,"
     f"`installtf`.")

STAGEDML_ROOT=environ.get('STAGEDML_ROOT')
if STAGEDML_ROOT is not None:
  assert isfile(join(STAGEDML_ROOT,'3rdparty','pylightnix','README.md')), \
      (f"Git submodule {join(STAGEDML_ROOT,'3rdparty','pylightnix')} looks uninitialized. "
       f"Did you run `git submodule update --init`?")

  assert isfile(join(STAGEDML_ROOT,'3rdparty','tensorflow_models','README.md')), \
      (f"Git submodule {join(STAGEDML_ROOT,'3rdparty','tensorflow_models')} looks uninitialized. "
       f"Did you run `git submodule update --init`?")
else:
  print(f"STAGEDML_ROOT env var is not set. We assume that you don't need development environment")


from pylightnix import PYLIGHTNIX_ROOT, PYLIGHTNIX_STORE, PYLIGHTNIX_TMP
if not isdir(PYLIGHTNIX_ROOT):
  print(
    f"Warning: PYLIGHTNIX_ROOT directory ('{PYLIGHTNIX_ROOT}') doesn`t exist. "
    f"Please create either direcotry or a symlink with this name.")

assert isdir(PYLIGHTNIX_STORE), \
    (f"PYLIGHTNIX_STORE directory ('{PYLIGHTNIX_STORE}') doesn`t exist. "
     f"Call `pylightnix.store_initialize` to initialize the storage from scratch" )

assert isdir(PYLIGHTNIX_TMP), \
    (f"PYLIGHTNIX_TMP directory ('{PYLIGHTNIX_TMP}') doesn`t exist. "
     f"Call `pylightnix.store_initialize` to initialize the storage from scratch" )


from stagedml.stages.fetchurl import WGET,AUNPACK

