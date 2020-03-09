#!/bin/sh

# Install the latest version of StagedML and Pylightnix
sudo -H make install

# Set PYTHONPATH to refer to 3rdparty modules, but use system StagedML and
# Pylighnitx. This way we could edit the sources while running a long-running task.
export PYTHONPATH=/workspace/3rdparty/nl2bash_essence/src:/workspace/3rdparty/tensorflow_models


run() {
VSIZE="$1"
python3 -c "
from pylightnix import realize, instantiate, redefine, mkconfig, promise
from stagedml.stages.all import *

def mysubtok(m):
  def _config(d):
    d['target_vocab_size']=$VSIZE
    d['vocab_file'] = [promise, 'vocab.%d' % $VSIZE]
    return mkconfig(d)
  return redefine(all_nl2bashsubtok,_config)(m)

def mytransformer(m):
  def _config(c):
    c['train_steps']=5*5000
    c['params']['beam_size']=3 # As in Tellina paper
    return mkconfig(c)
  return redefine(transformer_wmt,_config)(m, mysubtok(m))

realize(instantiate(mytransformer))
"
}

run 30000
run 25000
run 20000
run 15000
run 10000
run 5000
run 1000
run 500
