StagedML
========

StagedML brings manageability into Deep Learning by applying
[Nix](https://nixos.org/nix) ideas of software deployment to the domain of ML
model libraries. The project is currenlty focused on NLP models which often
require complex pre-processing and long training. StagedML uses minimalistic
immutable data engine named
[Pylightnix](https://github.com/stagedml/pylightnix).


Contents
--------

1. [Features](#Features)
2. [Install](#Install)
   - [System requirements](#system-requirements)
   - [Running the docker container](#running-docker-containers)
3. [Documentation](#documentation)
4. [Quick Start](#quick-start)


Features
--------

* StagedML is a library of adopted ML models. We do not claim any
  remarkable accuracy or performance achievements, but we do provide several
  infrastructure properties which simplify the development process.
  1. StagedML is powered by [Pylightnix](https://github.com/stagedml/pylightnix/)
     immutable data management library.
  2. All adopted models and datasets are defined as a linked graph of Pylightnix
     core objects called
     [stages](https://github.com/stagedml/pylightnix/blob/master/docs/Reference.md#pylightnix.types.Derivation).
     Dependency resolution is done automatically.
  3. Any stage could be created in one button click (here: by
     one line of Python code, not counting the imports). Example:
     ```python
     > from stagedml.stages.all import all_convnn_mnist, realize, instantiate, rref2path, shell
     > rref=realize(instantiate(all_convnn_mnist))
     # ^^^ Download the dataset, train the convolution network, etc..
     > rref
     'rref:2bf51e3ce37061ccff6168ccefac7221-3b9f88037f737f06af0fe82b6f6ac3c8-convnn-mnist'
     # ^^^ Realization reference describes a folder containing final checkpoints and training logs
     ```
  4. StagedML attempts to re-use already trained models whenever possible.
  5. For every stage, users could access it's full configuration, including the
     configurations of it's dependencies
     ```python
     > from pylightnix import mklens
     > mklens(rref).learning_rate.val   # Learning rate of the model
     0.001
     > mklens(rref).mnist.url.val       # URL of the dataset used to train the model
     'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
     ```
  6. StagedML **evaluates all the configurations before executing all the
     builders**. Thanks to this feature, equipped with
     [Lenses](https://github.com/stagedml/pylightnix/blob/master/docs/Reference.md#lens-objects) and
     [Promises](https://github.com/stagedml/pylightnix/blob/master/docs/Reference.md#pylightnix.types.PromisePath),
     we could catch configuration-time typos, misspelled parameter names and
     a large portion of incorrect paths before starting long training.
  7. Users could overwrite stage configurations by editing the source code!
     OK, sometimes we really can't avoid it. StagedML attempts to
     keep this process less painful:
     - Changing and running the code wouldn't overwrite any existing data.
     - Stages may often be re-wired on a higher-level without accessing
       low-level details.
     - In many cases we can tweak configurations in-place:
       ```python
       > from pylightnix import redefine
       > def _new_config(old_config):
       >   old_config.learning_rate = 1e-5
       >   return old_config
       > rref5=realize(instantiate(redefine(all_convnn_mnist, new_config=_new_config)))
       > rref5
       'rref:1ece593a8e761fa28fdc0da0fed00eb8-dd084d4a8b75a787b7c230474549e5db-convnn-mnist'
       > mklens(rref5).learning_rate.val
       1e-05
       ```
  8. Thanks to the
     [REPL API](https://github.com/stagedml/pylightnix/blob/master/docs/Reference.md#pylightnix.repl),
     we could debug intermediate stages by instructing Pylightnix to pause before
     starting certain constructors. The resulting procedure is similar to the one
     that we see in `git-rebase --continue` workflow. See also
     [REPL demo of Pylightnix](https://github.com/stagedml/pylightnix/blob/master/docs/demos/REPL.md).
  9. StagedML supports non-determenistic build processes which means that we
     could train several instances of the model and pick up the best one to use
     in subsequent stages. Selection criteria are up to the user. See `Matcher`
     topic of the Pylightnix documentation.
     ```python
     > rref2path(rref)
     '/tmp/pylightnix/store-v0/3b9f88037f737f06af0fe82b6f6ac3c8-convnn-mnist/2bf51e3ce37061ccff6168ccefac7221'
     # ^^^ Storage root        ^^^ Stage configuration                       ^^^ Stage realization (one of)
     ```
  10. Finally, StagedML offers basic garbage collector `stagedml.stages.all.gc`
      allowing users to keep the chosen set of stages (and thus all their
      dependencies) and remove the rest.

* Currently, we include some NLP models from
  [tensorflow-models](https://github.com/tensorflow/models), other libraries may
  be supported in future. Often we pick only BASE versions which could be
  trained on GPU. Check the [full collection of adopted
  models and datasets](./src/stagedml/stages/all.py)
* Deployment of trained models is not supported now but may be supported in
  future. Thanks to the simplicity of Pylightnix storage format, the deployment
  could probably be done just by running `rsync` on the Pylightnix storage
  folders of the source and target machines.
* StagedML is not tested as thoroughly as we wish it should. At
  the same time:
  - To minimize the uncertainty, we specify the exact versions of dependency
    libraries (TensorFlow, TensorFlow Models, Pylightnix) by linking them as Git
    submodules.
  - The considerable efforts were made to test the underlying Pylightnix
    library.
  - We extensively use [Mypy](http://mypy-lang.org/)-compatible type annotations.

Install
-------

The project is in its early stages, so we mainly focus on Docker-driven
environment for development and evaluation. We provide 2 docker containers, one
is defined by **[./docker/stagedml_dev.docker](./docker/stagedml_dev.docker)** and
describes development environment, another one is defined by
[./docker/stagedml_ci.docker](./docker/stagedml_ci.docker).  We plan to use it
for continuous integration tests in future.

### System requirements

* Linux system (other OS may accidently work too)
* GPU NVidia 1080Ti
* Docker with `nvidia-docker` runner and internet connection

### Running docker containers

We show how to run the project in development docker

1. Clone the Stagedml repo recursively
   ```
   $ git clone --recursive https://github.com/stagedml/stagedml
   ```

2. Cd to project's root and run the docker script to build the container.
   ```
   $ cd stagedml
   $ ./rundocker.sh
   ```

   The docker builder will download [deepo](https://github.com/ufoym/deepo) base
   image and additional dependencies. After the image is ready, the script will
   bind-mount project root folder as container's `/workspace`. Finally, it will
   open Bash shell with PYTHONPATH pointing to local Python sources, and a
   collection of [helper shell functions](./env.sh).

3. Now, we have to make sure we are using a compatible version of TensorFlow.
   At the time of this writing, the default TF from Deepo was a bit old, so we
   provide our favorite version as  `./3rdparty/tensorflow` Git submodule. You
   have the following options:

   * (a) Build our favorite TensorFlow from sources. Consider using some of our
     helper shell functions:

     ```
     $ buildtf
     ```

     That would take some time. The resulting `*wheel` will appear in `./_tf`
     folder. Once it is ready, call

     ```
     $ installtf
     ```

     to actually install the TF wheel into the container. Note, that you need to
     call `installtf` (but not `buildtf`) at each start of the container for now.

   * (b) Alternatively, you are free to experiment with any `tensorflow>=2.1`
     package from elsewhere. Just install it using `sudo -H pip3 install
     tensorflow-gpu` or `sudo apt-get install tensorflow-gpu`.

4. That is all. Run `ipython` to try StagedML in action.


Documentation
-------------

Not much yet:)

StagedML is a collection of Pylightnix stages, so
[Pylightnix documentation and manuals](https://github.com/stagedml/pylightnix/blob/master/README.md#Documentation)
do apply here:

* [MNIST demo](https://github.com/stagedml/pylightnix/blob/master/docs/demos/MNIST.md)
  shows the machine learning specifics of Pylightnix.
* [REPL demo](https://github.com/stagedml/pylightnix/blob/master/docs/demos/REPL.md)
  illustrates how to debug stages using Read-Eval-Print-friendly routines
  [(wiki)](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop).
* [Ultimatum tutorial](https://github.com/grwlf/ultimatum-game/blob/master/docs/Pylightnix.md)
  is a note on organizing experiments.
* [Pylightnix API Reference](https://github.com/stagedml/pylightnix/blob/master/docs/Reference.md)

Most of the stages are defined in [stagedml.stages](./src/stagedml/stages)
packages. The [stagedml.stages.all](./src/stagedml/stages/all.py) module contains
top-level definitions. Machine learning models are mostly borrowed from the
[TensorFlow Official Models](https://github.com/tensorflow/models), we keep
their main parts under the [stagedml.models](./src/stagedml/models)
module.

Below sections describe the typical work scenarios.

Quick Start
-----------

Top-level definitions are listed in a single
[all.py](./src/stagedml/stages/all.py) file.  There, every `all_` function
defines a *stage*, which is usually a model or a dataset. Every stage could be
built (or *realized*) by calling `realize(instantiate(...))` functions. Stages
may depend on each other and Pylightnix will manage dependencies automatically.

An example IPython session may look like the following:

```python
> from stagedml.stages.all import *
> # Initialize Pylightnix storage
> store_initialize()
> # Train the model of choice. Here - BERT with GLUE/MRPC task
> realize(instantiate(all_bert_finetune_glue, 'MRPC'))

# ..
# .. Download GLUE Dataset...
# .. Download pretrained BERT checkpoint
# .. Convert the Dataset into TFRecord format
# .. Fine tune the model
# ..

'rref:eedaa6f13fee251b9451283ef1932ca0-c32bccd3f671d6a3da075cc655ee0a09-bert'
```

Now, save the *realization reference* by typing `rref=_`. This reference could
be converted into storage folder with `pylightnix.rref2path` function.

```python
> print(rref2path(rref))
/var/run/pylightnix/store-v0/c32bccd3f671d6a3da075cc655ee0a09/eedaa6f13fee251b9451283ef1932ca0/
```

With the realization reference in hands, we could:

- Examine training logs and figures by accessing training artifacts located in
  storage folder returned by `pylightnix.rref2path`.
- Obtain derivation reference with `pylightnix.rref2dref` and define new stages
  based on existing stages. StagedML tracks configurations and prevent you from
  messing up the data.
- Run TensorBoard by passing RRef to `stagedml.utils.tf.runtb`. Also we have
  `stagedml.core.tbrealize` function to run TensorBoard at the time of stage's
  realization. Assuming that we run StagedML in Docker as described in the
  Install section, we could run `./runchrome.sh` script from Host machine to
  connect a web-client and monitor the training process.
- Tweak model parameters with `pylightnix.redefine`, re-train the model while
  keeping results of previous trainings.
- Run the garbage collector `stagedml.stages.all.gc` to remove outdated data.

