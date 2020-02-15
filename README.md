Stagedml
========

Stagedml brings manageability into Deep Learning by applying
[Nix](https://nixos.org/nix) ideas of software deployment.
Currently the project is focused on NLP models which often requre
complex pre-processing and long training. Stagedml uses minimalistic
immutable data engine named
[Pylightnix](https://github.com/stagedml/pylightnix) under the hood.


Contents
--------

1. [Features](#Features)
2. [Install](#Install)
   - [System requirements](#system-requirements)
   - [Running the docker container](#running-docker-containers)
3. [Working with Stagedml](#working-with-stagedml)


Features
--------

* Stagedml is a library of adopted ML models, focused on the ease of
  experimenting.
* Currently, it includes some TensorFlow NLP models from
  [tensorflow-models](https://github.com/tensorflow/models), other libraries may
  be supported in future.
* All addopted models are defined as a linked graph of [Pylightnix
  stages](https://github.com/stagedml/pylightnix/blob/master/docs/Reference.md#pylightnix.types.Derivation).
  Dependency resolution is done automatically.
* Datasets and Model checkpoints are cached and hashed into the filesystem
  storage.
* We extensively use [Mypy](http://mypy-lang.org/)-compatible type annotations.
* Check the list of [adopted models and datasets](./src/stagedml/stages/all.py)

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

Working with StagedML
---------------------

Stagedml is desinged as a [Nix](https://nixos.org/nix)-style collection of
ML models.

Main top-level definitions are collected in a single
[all.py](./src/stagedml/stages/all.py) file.  In this file, each `all_`
function defines a *stage*, which is usually a model or a dataset. Every stage
could be *realized* by calling `realize(instantiate(stage))` functions. Stages
may depend on each other and Pylightnix' core will manage dependencies
automatically.

So, an example IPython session could look like the following:

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

In general, With realization reference in hands, you could:

- Examine training logs and figures by accessing training artifacts located in
  storage folder returned by `rref2path`.
- Change model parameters and re-train it without loosing previous results
  (until Pylightnix garbage collection is run).
- Build new models based on the current model's checkpoints. Stagedml will track
  stage configurations and prevent you from messing up the data.
- (TODO) Run Pylightnix garbage collector to remove unused models.

