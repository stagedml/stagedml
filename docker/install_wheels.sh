#!/bin/sh

set -e -x
for whl in /install/wheels/* ; do
  pip3 install --force $whl
done

pip3 install tensorflow_hub
