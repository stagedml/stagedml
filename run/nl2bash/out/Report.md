NL2Bash experiment
==================


```python
from stagedml.imports import environ, join, makedirs
O=join(environ['STAGEDML_ROOT'], '_pylightnix', 'experiments', 'nl2bash2')
makedirs(O, exist_ok=True)
```




```python
from pylightnix import (
    Path, realize, instantiate, redefine, mkconfig, promise, rref2dref,
    mksymlink )
from stagedml.stages.all import *
from logging import getLogger

getLogger('tensorflow').setLevel('INFO')

def run(vsize:int)->Path:
  def mysubtok(m):
    def _config(d):
      d['target_vocab_size']=vsize
      d['vocab_file'] = [promise, 'vocab.%d' % vsize]
      d['train_data_min_count']=None
      d['file_byte_limit'] = 1e6 if vsize > 5000 else 1e5
      return mkconfig(d)
    return redefine(all_nl2bashsubtok,_config)(m)

  def mytransformer(m):
    def _config(c):
      c['train_steps']=6*5000
      c['params']['beam_size']=3 # As in Tellina paper
      return mkconfig(c)
    return redefine(transformer_wmt,_config)(m, mysubtok(m))

  rref=realize(instantiate(mytransformer))
  return mksymlink(rref, O, vsize, withtime=False)
```





```python
run(15000)
```

```
'/workspace/_pylightnix/experiments/nl2bash2/15000'
```




Print?
Hiiiiiii1

Print?
Hiiiiii2
