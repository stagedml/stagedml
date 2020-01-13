Stagedml
========

Stagedml brings manageability into Deep Learning by applying
[Nix](https://nixos.org/nix) ideas of software deployment to new domain.
Currently it is focused on TensorFlow NLP models which requries long multi-staged
training. Under the hood Stagedml uses minimalistic engine called
[pylightnix](https://github.com/stagedml/pylightnix) for storage management.

Install
-------

The project is in its early stages, so we mainly focus on Docker-driven
environment for development and evaluation. We provide 2 docker containers, one
is defined by [./docker/stagedml_dev.docker](./docker/stagedml_dev.docker) and
describes development environment, another one is defined by
[./docker/stagedml_ci.docker](./docker/stagedml_ci.docker).  We plan to use it
for continuous integration tests.

### System requirements

* Linux system (other OS may accidently work too)
* GPU NVidia 1080Ti
* Docker with `nvidia-docker` runner

### Running docker containers

We show how to run the project in development docker

1. Clone the repo recursively
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
   open Bash shell where PYTHONPATH points to local sources and
   [several helper functions](./env.sh) are defined.

3. Now, we have to make sure we are using a compatible version of TensorFlow.
   At the time of this writing, the default TF from Deepo was a bit old, so we
   provide our favorite version as  `./3rdparty/tensorflow` submodule. In order
   to use it, we have to build it first.  Consider using our helper function

   ```
   $ buildtf
   ```

   The resulting `*wheel` will appear in `./_tf` folder. Once it is ready, call

   ```
   $ installtf
   ```

   to actually install it into the container. Note, that you need to call
   `installtf` at every start of the container for now.

4. Alternatively, you are free to experiment with any `tensorflow-2.1`
   package from elsewhere. Just install it using `sudo -H pip3 install` or `sudo apt-get install`.

5. That is all. Now you could run `ipython` to call functions directly or run
   scripts from `./run` folder.

