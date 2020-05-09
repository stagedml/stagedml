#!/bin/sh

# Exit on first error and echo commands to stderr
set -e -x

# Create a folder which would contain linked results of this experiment
test -d "$STAGEDML_ROOT"
O="$STAGEDML_ROOT/_pylightnix/experiments/nl2bash2"
mkdir -p "$O" | true

# Install the latest version of Pylightnix, TensorFlow models, StagedML into the
# system using Pip. Needs sudo here once.
sudo -H make install

# From this point, use system-wide Python packages. This way we could edit the
# sources while doing long-running tasks in the background.
unset PYTHONPATH

run() {
VSIZE="$1"
python3 -c "
from pylightnix import ( realize, instantiate, redefine, mkconfig, promise,
                         rref2dref, mksymlink )
from stagedml.stages.all import *
from logging import getLogger

getLogger('tensorflow').setLevel('INFO')

def mysubtok(m):
  def _config(d):
    d['target_vocab_size']=$VSIZE
    d['vocab_file'] = [promise, 'vocab.%d' % $VSIZE]
    d['train_data_min_count']=None
    d['file_byte_limit'] = 1e6 if $VSIZE > 5000 else 1e5
  return redefine(all_nl2bashsubtok,_config)(m)

def mytransformer(m):
  def _config(c):
    c['train_steps']=6*5000
    c['params']['beam_size']=3 # As in Tellina paper
  return redefine(transformer_wmt,_config)(m, mysubtok(m))

rref=realize(instantiate(mytransformer))
mksymlink(rref, '$O', str('$VSIZE'), withtime=False)
"
}

run 15000
run 10000
run 5000
run 1700

python3 -c "from stagedml.utils.tf import runtb; runtb('$O')"

