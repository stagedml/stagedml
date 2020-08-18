#!/bin/sh
conda install -c conda-forge \
  tqdm \
  hypothesis \
  pytest \
  ipdb \
  pyls-mypy \
  mypy \
  pytest-mypy \
  codecov \
  coverage \
  pygraphviz \
  matplotlib \
  pyqt \
  pweave && \
pip3 install \
  git+https://github.com/stagedml/pydoc-markdown.git@develop \
  netron \
  beautifultable
