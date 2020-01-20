#!/bin/sh
pip3 install --upgrade pip && \
pip3 install seaborn PyQT5 PySide2 \
             beautifultable tqdm \
             hypothesis pytest \
             seqeval sklearn \
             jsonpickle overrides \
             ipdb netron \
             pyls pyls-mypy mypy \
             pytest-mypy \
             git+https://github.com/stagedml/pydoc-markdown.git@develop

