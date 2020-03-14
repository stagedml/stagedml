NL2Bash experiment
==================

This document describes the results of applying Official Trnasformer of the
TensorFlow ([link][1])
to the NL2Bash dataset described in the [NL2Bash: A Corpus and Semantic Parser
for Natural Language Interface to the Linux Operating
System][2].

Intro
-----

The primary goal of this work is to demonstrate the features of [StagedML
library][3]. The secondary goal is to evaluate the
[NL2BASH](https://github.com/stagedml/nl2bash_essence/tree/master/src/data/bash)
dataset.

In particular, we want to highlight the following facts:

1. The sinppets of top-level code are only one-screen long. At the same time, it
   allows the programmer to change every aspect of the experiment by connecting
   stages together and tweaking their parameters.
2. StagedML caches the results of running stages by creating immutable objects
   in the Pylightnix storage. Configurations and training results are
   accessible to users via [Pylightnix][4] API.
3. Consequently, the rendering of the experiment reports may be largely authomatized.
   In this work, we generated reports from scrath by running a `Makefile` in the
   [/run/nl2bash](/run/nl2bash) directory.

The source files of this report are
[available](https://github.com/stagedml/stagedml/tree/master/run/nl2bash/Report.md.in)
in StagedML repository. We used [PWeave](http://mpastell.com/pweave) to render
this report.

### The Model

In this work, we trained an close copy of official Transformer of the
TensorFlow, using the [NL2BASH
dataset](https://github.com/stagedml/nl2bash_essence/tree/master/src/data/bash)
inroduced by the [NL2BASH paper][2].

The TensorFlow code of the model is located in the
[transformer_wmt.py](/src/stagedml/stages/transformer_wmt.py) file. The model
consists of the `TransformerBuild` class for storing the mutable state and of
the number of operations, including `build`, `train`, `evaluate` and others.
Every operation typically accepts `TransformerBuild` object and the index of the
model instance.

Finally, we define Pylightnix stage in the `transformer_wmt` function. It's
arguments are:

- `m` Pylightnix dependency resolution context.
- `wmt:WmtSubtok` reference to the upstream stage providing a tokenizer and a
  dataset
- `num_instances:int=1` the number of instances to train.

### Metrics

To report the model performance we used BLEU metrics, as implemented in
TensorFlow official Trnasformer model. This metric may differs from the version
of BLEU which were used by the authors of NL2BASH paper[2][2], so we can't
compare results directly.

We applied the metrics to the evaluation subset of the NL2Bash dataset which is
a `0.1` part of the original dataset.

### Experiments


1. [Baseline Transformer](#baseline-transformer) -  use default upstream settings
2. [Unshuffled Transformer](#unshuffled-transformer) -  don't shuffle the dataset
3. [Bash specific tokens](#bash-specific-tokens) - add all commands and flags to
   the list of subtokens
4. [Changing vocabulary size](#changing-vocabulary-size) - try different target
   sizes of vocabulary
5. [Single-char punctuation tokens](#single-char-punctuation-tokens) - suppress
   multichar punktuation subtokens

Imports
-------


```python
import numpy as np
import matplotlib.pyplot as plt

from logging import getLogger

from pylightnix import (
    RRef, Path, realize, instantiate, redefine, mkconfig, promise, rref2dref,
    mksymlink, rref2path, mklens )

from stagedml.imports import environ, join, environ, makedirs
from stagedml.stages.all import transformer_wmt, all_nl2bashsubtok
from analyze import read_tensorflow_log

getLogger('tensorflow').setLevel('INFO')
```



Baseline transformer
--------------------

Below we define the top-level code which sets task-specific parameters of the
`transformer_wmt` stage.


```python
from pylightnix import ( RRef, realizeMany, instantiate, redefine, mkconfig,
    mksymlink, match_some )
from stagedml.stages.all import transformer_wmt, all_nl2bashsubtok

def baseline_subtok(m):
  return all_nl2bashsubtok(m, shuffle=True,
                              with_bash_charset=False,
                              with_bash_subtokens=False)

def baseline_transformer(m):
  def _config(c):
    c['train_steps']=6*5000
    c['params']['beam_size']=3 # As in Tellina paper
    return mkconfig(c)
  return redefine(transformer_wmt,
                  new_config=_config,
                  new_matcher=match_some())(m, baseline_subtok(m), num_instances=5)
```



### Model size

The size of the model is not saved as a specieal part of the model output, but
we have defined a small function which reads it via Keras API.


```python
from analyze import model_size
```



The number of trainable weights of baseline model is
---------------------------------------------------------------------------NameError                                 Traceback (most recent call last)<ipython-input-1-1a49f203f0af> in <module>
----> 1 model_size(baseline_transformer)
NameError: name 'model_size' is not defined parameters.


### Evaluation

We now display the BLEU metrics of the model during first 6 training epoches.


```python
plt.figure(1)
plt.xlabel("Training steps")
plt.title("BLEU-cased, Baseline transformer")

out=join(environ['STAGEDML_ROOT'],'_experiments','nl2bash0')
makedirs(out, exist_ok=True)
for i,rref in enumerate(realizeMany(instantiate(baseline_transformer))):
  mksymlink(rref, out, f'run-{i}', withtime=False)
  baseline_bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
  plt.plot(range(len(baseline_bleu)), baseline_bleu, label=f'run-{i}', color='blue')

plt.legend(loc='upper left', frameon=True)
plt.grid(True)
```

![](figures/Report.md_figure4_1.png)\


Unshuffled dataset
------------------

This experiment


```python
from pylightnix import RRef, realize, instantiate, redefine, mkconfig, mksymlink
from stagedml.stages.all import transformer_wmt, all_nl2bashsubtok

def runU(out)->RRef:
  def mysubtok(m):
    return all_nl2bashsubtok(m, shuffle=False,
                                with_bash_charset=False,
                                with_bash_subtokens=False)

  def mytransformer(m):
    def _config(c):
      c['train_steps']=6*5000
      c['params']['beam_size']=3 # As in Tellina paper
      return mkconfig(c)
    return redefine(transformer_wmt,_config)(m, mysubtok(m))

  rref=realize(instantiate(mytransformer))
  makedirs(out, exist_ok=True)
  mksymlink(rref, out, 'result', withtime=False)
  return rref
```




```python
plt.figure(1)
plt.xlabel("Training steps")
plt.title("BLEU-cased")

rref=runU(join(environ['STAGEDML_ROOT'],'_experiments','nl2bash0'))
unshuffled_bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
plt.plot(range(len(unshuffled_bleu)), unshuffled_bleu, label=f'Unshuffled transformer')

plt.legend(loc='upper left', frameon=True)
plt.grid(True)
```

![](figures/Report.md_figure6_1.png)\



Bash-specific tokens
--------------------

We define the model running funtion `run1` using the following StagedML
_stages_:

 * `nl2bashsubtok` which performs the sub-tokenization.
 * `transformer_wmt` where the Transformer model is defined. The model is very
   similar to the Official Transformer of TensorFlow.

`nl2bashsubtok` stage uses lower-level stages like `fetchnl2bash` but we hide
this fact by using top-level wrapper `all_nl2bashsubtok`.


```python
def run1(vsize:int)->RRef:
  def mysubtok(m):
    def _config(d):
      d['target_vocab_size']=vsize
      d['vocab_file'] = [promise, 'vocab.%d' % vsize]
      return mkconfig(d)
    return all_nl2bashsubtok(m, shuffle=True,
                                with_bash_charset=True,
                                with_bash_subtokens=True)

  def mytransformer(m):
    def _config(c):
      c['train_steps']=5*5000
      c['params']['beam_size']=3 # As in Tellina paper
      return mkconfig(c)
    return redefine(transformer_wmt,_config)(m, mysubtok(m))

  return realize(instantiate(mytransformer))
```




The model's parameters are


```python
from pprint import PrettyPrinter
PrettyPrinter(indent=4, compact=True).pprint(mklens(run1(15000)).as_dict())
```





```python
plt.figure(2)
plt.xlabel("Training steps")
plt.title("BLEU")

plt.plot(range(len(unshuffled_bleu)), unshuffled_bleu, label=f'Unshuffled transformer', color='red')
plt.plot(range(len(baseline_bleu)), baseline_bleu, label=f'Baseline transformer', color='orange')

for i,vsize in enumerate([ 30000, 25000, 20000, 15000, 10000, 5000, 1000, 500 ]) :
  rref=run1(vsize)
  bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
  plt.plot(range(len(bleu)), bleu, label=f'run-{i}', color='blue')

plt.legend(loc='upper left', frameon=True)
plt.grid(True)
```

```
Building Transformer instance 1 of 1
Setting vocab_size to 8022
```

```
---------------------------------------------------------------------------ResourceExhaustedError
Traceback (most recent call last)<ipython-input-1-bc7d59b89718> in
<module>
      7
      8 for i,vsize in enumerate([ 30000, 25000, 20000, 15000, 10000,
5000, 1000, 500 ]) :
----> 9   rref=run1(vsize)
     10   bleu=read_tensorflow_log(join(rref2path(rref),'eval'),
'bleu_cased')
     11   plt.plot(range(len(bleu)), bleu, label=f'run-{i}',
color='blue')
<ipython-input-1-56a7f2a3812b> in run1(vsize)
     16     return redefine(transformer_wmt,_config)(m, mysubtok(m))
     17
---> 18   return realize(instantiate(mytransformer))
/usr/local/lib/python3.6/dist-packages/pylightnix/core.py in
realize(closure, force_rebuild, assert_realized)
    731   """ A simplified version of
[realizeMany](#pylightnix.core.realizeMany).
    732   Expects only one output path. """
--> 733   rrefs=realizeMany(closure, force_rebuild, assert_realized)
    734   assert len(rrefs)==1, (
    735       f"`realize` is to be used with single-output
derivations. Derivation "
/usr/local/lib/python3.6/dist-packages/pylightnix/core.py in
realizeMany(closure, force_rebuild, assert_realized, realize_args)
    796                            assert_realized=assert_realized,
    797                            realize_args=realize_args)
--> 798     next(gen)
    799     while True:
    800       gen.send((None,False)) # Ask for default action
/usr/local/lib/python3.6/dist-packages/pylightnix/core.py in
realizeSeq(closure, force_interrupt, assert_realized, realize_args)
    837             f"{store_config(dref)}"
    838             )
--> 839
paths=drv.realizer(dref,dref_context,realize_args.get(dref,{}))
    840           rrefs_built=[store_realize(dref,dref_context,path)
for path in paths]
    841           rrefs_matched=drv.matcher(dref,dref_context)
/usr/local/lib/python3.6/dist-packages/pylightnix/core.py in
_realizer(dref, ctx, rarg)
    662   def _promise_aware(realizer)->Realizer:
    663     def
_realizer(dref:DRef,ctx:Context,rarg:RealizeArg)->List[Path]:
--> 664       outpaths=realizer(dref,ctx,rarg)
    665       for key,refpath in
config_promises(store_config_(dref),dref):
    666         for o in outpaths:
/usr/local/lib/python3.6/dist-packages/pylightnix/core.py in
_realizer(dref, ctx, rarg)
    662   def _promise_aware(realizer)->Realizer:
    663     def
_realizer(dref:DRef,ctx:Context,rarg:RealizeArg)->List[Path]:
--> 664       outpaths=realizer(dref,ctx,rarg)
    665       for key,refpath in
config_promises(store_config_(dref),dref):
    666         for o in outpaths:
/usr/local/lib/python3.6/dist-packages/pylightnix/core.py in
_wrapper(dref, context, rarg)
    443   def _wrapper(dref,context,rarg)->List[Path]:
    444     timeprefix=timestring() if buildtime else None
--> 445     b=ctr(mkbuildargs(dref,context,timeprefix,{},rarg)); f(b);
return list(getattr(b,'outpaths'))
    446   return _wrapper
    447
/usr/local/lib/python3.6/dist-
packages/stagedml/stages/transformer_wmt.py in _realize(b)
    188     for instance_idx in range(num_instances):
    189       print(f'Building Transformer instance {instance_idx+1}
of {num_instances}')
--> 190       build(b,instance_idx)
    191       train(b,instance_idx)
    192
/usr/local/lib/python3.6/dist-
packages/stagedml/stages/transformer_wmt.py in build(b, instance_idx)
     64   set_session_config(enable_xla=c.enable_xla)
     65
---> 66   b.train_model = create_train_model(c.params)
     67   b.train_model.compile(create_optimizer(c.params))
     68   b.train_model.summary()
/usr/local/lib/python3.6/dist-
packages/stagedml/models/transformer/model.py in
create_train_model(params)
     21     inputs = tf.keras.layers.Input((None,), dtype="int64",
name="inputs")
     22     targets = tf.keras.layers.Input((None,), dtype="int64",
name="targets")
---> 23     internal_model = TransformerLayer(params,
name='transformerv2')
     24     logits = internal_model([inputs, targets], training=True)
     25     vocab_size = params["vocab_size"]
/usr/local/lib/python3.6/dist-
packages/stagedml/models/transformer/model.py in __init__(self,
params, **kwargs)
     65     self.params = params
     66     self.embedding_softmax_layer = EmbeddingSharedWeights(
---> 67         params["vocab_size"], params["hidden_size"])
     68     self.encoder_stack = EncoderStack(params)
     69     self.decoder_stack = DecoderStack(params)
/usr/local/lib/python3.6/dist-
packages/stagedml/models/transformer/embedding.py in __init__(self,
vocab_size, hidden_size)
     28           shape=[self.vocab_size, self.hidden_size],
     29           initializer=tf.random_normal_initializer(
---> 30               mean=0., stddev=self.hidden_size**-0.5))
     31       # self.shared_weights = tf.Variable(
     32       #     name="embedding_and_softmax_weights",
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/keras/engine/base_layer.py in
add_weight(self, name, shape, dtype, initializer, regularizer,
trainable, constraint, partitioner, use_resource, synchronization,
aggregation, **kwargs)
    444         synchronization=synchronization,
    445         aggregation=aggregation,
--> 446         caching_device=caching_device)
    447     backend.track_variable(variable)
    448
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/training/tracking/base.py in
_add_variable_with_custom_getter(self, name, shape, dtype,
initializer, getter, overwrite, **kwargs_for_getter)
    742         dtype=dtype,
    743         initializer=initializer,
--> 744         **kwargs_for_getter)
    745
    746     # If we set an initializer and the variable processed it,
tracking will not
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/keras/engine/base_layer_utils.py in
make_variable(name, shape, dtype, initializer, trainable,
caching_device, validate_shape, constraint, use_resource, collections,
synchronization, aggregation, partitioner)
    140       synchronization=synchronization,
    141       aggregation=aggregation,
--> 142       shape=variable_shape if variable_shape else None)
    143
    144
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/variables.py in __call__(cls,
*args, **kwargs)
    256   def __call__(cls, *args, **kwargs):
    257     if cls is VariableV1:
--> 258       return cls._variable_v1_call(*args, **kwargs)
    259     elif cls is Variable:
    260       return cls._variable_v2_call(*args, **kwargs)
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/variables.py in
_variable_v1_call(cls, initial_value, trainable, collections,
validate_shape, caching_device, name, variable_def, dtype,
expected_shape, import_scope, constraint, use_resource,
synchronization, aggregation, shape)
    217         synchronization=synchronization,
    218         aggregation=aggregation,
--> 219         shape=shape)
    220
    221   def _variable_v2_call(cls,
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/variables.py in <lambda>(**kwargs)
    195                         shape=None):
    196     """Call on Variable class. Useful to force the
signature."""
--> 197     previous_getter = lambda **kwargs:
default_variable_creator(None, **kwargs)
    198     for _, getter in
ops.get_default_graph()._variable_creator_stack:  # pylint:
disable=protected-access
    199       previous_getter = _make_getter(getter, previous_getter)
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/variable_scope.py in
default_variable_creator(next_creator, **kwargs)
   2594         synchronization=synchronization,
   2595         aggregation=aggregation,
-> 2596         shape=shape)
   2597   else:
   2598     return variables.RefVariable(
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/variables.py in __call__(cls,
*args, **kwargs)
    260       return cls._variable_v2_call(*args, **kwargs)
    261     else:
--> 262       return super(VariableMetaclass, cls).__call__(*args,
**kwargs)
    263
    264
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/resource_variable_ops.py in
__init__(self, initial_value, trainable, collections, validate_shape,
caching_device, name, dtype, variable_def, import_scope, constraint,
distribute_strategy, synchronization, aggregation, shape)
   1409           aggregation=aggregation,
   1410           shape=shape,
-> 1411           distribute_strategy=distribute_strategy)
   1412
   1413   def _init_from_args(self,
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/resource_variable_ops.py in
_init_from_args(self, initial_value, trainable, collections,
caching_device, name, dtype, constraint, synchronization, aggregation,
distribute_strategy, shape)
   1540           with ops.name_scope("Initializer"),
device_context_manager(None):
   1541             initial_value = ops.convert_to_tensor(
-> 1542                 initial_value() if init_from_fn else
initial_value,
   1543                 name="initial_value", dtype=dtype)
   1544           if shape is not None:
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/keras/engine/base_layer_utils.py in
<lambda>()
    120           (type(init_ops.Initializer),
type(init_ops_v2.Initializer))):
    121         initializer = initializer()
--> 122       init_val = lambda: initializer(shape, dtype=dtype)
    123       variable_dtype = dtype.base_dtype
    124   if use_resource is None:
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/init_ops_v2.py in __call__(self,
shape, dtype)
    282     dtype = _assert_float_dtype(dtype)
    283     return self._random_generator.random_normal(shape,
self.mean, self.stddev,
--> 284                                                 dtype)
    285
    286   def get_config(self):
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/init_ops_v2.py in
random_normal(self, shape, mean, stddev, dtype)
    777       op = random_ops.random_normal
    778     return op(
--> 779         shape=shape, mean=mean, stddev=stddev, dtype=dtype,
seed=self.seed)
    780
    781   def random_uniform(self, shape, minval, maxval, dtype):
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/random_ops.py in
random_normal(shape, mean, stddev, dtype, seed, name)
     72     seed1, seed2 = random_seed.get_seed(seed)
     73     rnd = gen_random_ops.random_standard_normal(
---> 74         shape_tensor, dtype, seed=seed1, seed2=seed2)
     75     mul = rnd * stddev_tensor
     76     value = math_ops.add(mul, mean_tensor, name=name)
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/ops/gen_random_ops.py in
random_standard_normal(shape, dtype, seed, seed2, name)
    639         pass  # Add nodes to the TensorFlow graph.
    640     except _core._NotOkStatusException as e:
--> 641       _ops.raise_from_not_ok_status(e, name)
    642   # Add nodes to the TensorFlow graph.
    643   dtype = _execute.make_type(dtype, "dtype")
/usr/local/lib/python3.6/dist-
packages/tensorflow_core/python/framework/ops.py in
raise_from_not_ok_status(e, name)
   6604   message = e.message + (" name: " + name if name is not None
else "")
   6605   # pylint: disable=protected-access
-> 6606   six.raise_from(core._status_to_exception(e.code, message),
None)
   6607   # pylint: enable=protected-access
   6608
~/.local/lib/python3.6/site-packages/six.py in raise_from(value,
from_value)
ResourceExhaustedError: OOM when allocating tensor with
shape[8022,512] and type float on
/job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
[Op:RandomStandardNormal]
```

![](figures/Report.md_figure9_1.png)\



Changing vocabulary size
------------------------

Customisations touch the Subtokenizer which were used to split both the input
sentences and output Bash commands into subtokens. They are:

1. We changed the Master Character Set of the Subtokenizer by adding
   `['-','+',',','.']` to the default list of Alphanumeric characters
2. We pre-parsed the train part of BASH dataset and generated the list of
   pre-defined subtokens. We include there:
    - First words of every sentences. Often those are command names.
    - All words starting from `-`. Often those are flags of bash commands.
3. We set the target size of the subtoken vocabulary to different values in
   range `[1000, 15000]`. The results are reported below.



```python
def run2(vsize:int, out:str)->None:
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
  makedirs(out, exist_ok=True)
  mksymlink(rref, out, vsize, withtime=False)
  return rref
```




```python
plt.figure(3)
plt.xlabel("Training steps")
plt.title("BLEU")

plt.plot(range(len(unshuffled_bleu)), unshuffled_bleu, label=f'Unshuffled transformer', color='red')
plt.plot(range(len(baseline_bleu)), baseline_bleu, label=f'Baseline transformer', color='orange')

for i,vsize in enumerate([ 15000, 10000, 5000, 1700 ]) :
  rref=run2(vsize, join(environ['STAGEDML_ROOT'],'_experiments','nl2bash2'))
  bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
  plt.plot(range(len(bleu)), bleu, label=f'vsize-{vsize}', color='blue')

plt.legend(loc='upper left', frameon=True)
plt.grid(True)
```

![](figures/Report.md_figure11_1.png)\


Single-char punctuation tokens
------------------------------


```python
from pylightnix import match_some, realizeMany, instantiate, redefine, mkconfig, mksymlink
from stagedml.stages.all import transformer_wmt, all_nl2bashsubtok
```



In addition to the modifications that we made in Experiment 2, we now attempt to
force the tokenizer to produce single-char tokens for punctuation.


```python
def exp3_mysubtok(m, vsize=10000):
  def _config(d):
    d['target_vocab_size']=vsize
    d['vocab_file'] = [promise, 'vocab.%d' % vsize]
    d['no_slave_multichar'] = True
    d['train_data_min_count']=None
    return mkconfig(d)
  return redefine(all_nl2bashsubtok,_config)(m)
```





```python
def exp3_mytransformer(m):
  def _config(c):
    c['train_steps']=6*5000
    c['params']['beam_size']=3 # As in Tellina paper
    return mkconfig(c)
  return redefine(transformer_wmt,
                  new_config=_config,
                  new_matcher=match_some())(m, exp3_mysubtok(m), num_instances=5)
```



Results:


```python
plt.figure(4)
plt.xlabel("Single-char punctuation tokens")
plt.title("BLEU")

plt.plot(range(len(unshuffled_bleu)), unshuffled_bleu, label=f'Unshuffled transformer', color='red')
plt.plot(range(len(baseline_bleu)), baseline_bleu, label=f'Baseline transformer', color='orange')

out=join(environ['STAGEDML_ROOT'],'_experiments','nl2bash3')
makedirs(out, exist_ok=True)
for i,rref in enumerate(realizeMany(instantiate(exp3_mytransformer))):
  mksymlink(rref, out, f'run-{i}', withtime=False)
  bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
  plt.plot(range(len(bleu)), bleu, label=f'run-{i}', color='blue')

plt.legend(loc='upper left', frameon=True)
plt.grid(True)
```

![](figures/Report.md_figure15_1.png)\


Unfortunately, suppressing punktuation seems to have no effect or the effect is
negative.


References
----------

[1]: https://github.com/tensorflow/models/tree/master/official/nlp/transformer
[2]: https://arxiv.org/abs/1802.08979
[3]: https://github.com/stagedml/stagedml
[4]: https://github.com/stagedml/pylightnix
