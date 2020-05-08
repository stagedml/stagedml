#!/bin/sh

set -e -x

case $1 in
  "") for whl in /install/wheels/* ; do
        pip3 install --force $whl
      done
      pip3 install tensorflow_hub
      ;;
  tensorflow|tf)
      pip3 install --force /install/wheels/tensorflow*
      pip3 install tensorflow_hub
      ;;
  *)  pip3 install --force /install/wheels/$1*
      ;;
esac
