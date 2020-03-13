#!/bin/sh

# Exit on first error and echo commands to stderr
set -e -x

# Create a folder which would contain linked results of this experiment
test -d "$STAGEDML_ROOT"
O="$STAGEDML_ROOT/_pylightnix/experiments/nl2bash1"
mkdir -p "$O" | true

# Install the latest version of StagedML and Pylightnix
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

rref=realize(instantiate(mytransformer))
mksymlink(rref, '$O', str('$VSIZE'), withtime=False)
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

python3 -c "from stagedml.utils.tf import runtb; runtb('$O')"

