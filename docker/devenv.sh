if test -z "$CI_BUILD_HOME" ; then
  echo "devenv.sh requires CI_BUILD_HOME to be set" >&2
  CI_BUILD_HOME=`pwd`
fi

export STAGEDML_SOURCE="$CI_BUILD_HOME"
export STAGEDML_ROOT="$CI_BUILD_HOME"
export STAGEDML_EXPERIMENTS="$STAGEDML_ROOT/_experiments"
export STAGEDML_RUSENTIMENT="$STAGEDML_ROOT/3rdparty/rusentiment.tar.xz"
export TERM=xterm-256color # TODO: check and document
export PATH="$STAGEDML_ROOT/.nix_docker_inject.env/bin:$STAGEDML_ROOT/3rdparty/wikiextractor:$PATH"
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude 3rdparty --exclude .git'
export PYTHONPATH=""
export MYPYPATH=""

update_pythonpath() {
  p="$1"
  if test -d "$p" ; then
    export PYTHONPATH="$p:$PYTHONPATH"
    export MYPYPATH="$p:$MYPYPATH"
    echo "Directory '$p' added to PYTHONPATH" >&2
  else
    echo "Directory '$p' doesn't exists. Not adding to PYTHONPATH" >&2
  fi
}

for p in \
  $STAGEDML_ROOT/3rdparty/tensorflow_models \
  $STAGEDML_ROOT/3rdparty/pylightnix/src \
  $STAGEDML_ROOT/3rdparty/nl2bash_essence/src \
  $STAGEDML_ROOT/3rdparty/keras-* \
  $STAGEDML_ROOT/src \
  ; do
  update_pythonpath "$p"
done

if test -f $HOME/.bashrc ; then
  if test -z "$STAGEDML_WANT_BASHRC" ; then
    STAGEDML_WANT_BASHRC=y
    . $HOME/.bashrc
  fi
fi

alias ipython="sh $STAGEDML_ROOT/ipython.sh"

if test -f $STAGEDML_ROOT/.nix_docker_inject.env/etc/myprofile ; then
  . $STAGEDML_ROOT/.nix_docker_inject.env/etc/myprofile
fi


runjupyter() {
  jupyter-notebook --ip 0.0.0.0 --port 8888 \
    --NotebookApp.token='' --NotebookApp.password='' "$@" --no-browser
}

runtensorboard() {(
  mkdir "$STAGEDML_ROOT/_logs" || true
  echo "Tensorboard logs redirected to $STAGEDML_ROOT/_logs/tensorboard.log"
  if test -n "$1" ; then
    args="--logdir $1"
    shift
  else
    args="--logdir $STAGEDML_ROOT/_logs"
  fi
  killall tensorboard
  tensorboard --host 0.0.0.0 $args "$@" >"$STAGEDML_ROOT/_logs/tensorboard.log" 2>&1
) & }

runchrome() {(
  mkdir -p "$HOME/.chrome_stagedml" || true
  chromium \
    --user-data-dir="$HOME/.chrome_stagedml" \
    --explicitly-allowed-ports=`seq -s , 6000 1 6020`,`seq -s , 8000 1 8020`,7777 \
    http://127.0.0.1:`expr 6000 + $UID - 1000` "$@"
)}

runnetron() {
  netron --host 0.0.0.0 -p 6006 "$@"
}

buildtfa() {(
  set -e -x
  cd $STAGEDML_ROOT/3rdparty/tensorflow_addons/

  export TF_NEED_CUDA="1"
  case $CUDA_VERSION in
    10.0*) export TF_CUDA_VERSION="10.0" ;;
    10.1*) export TF_CUDA_VERSION="10.1" ;;
    *) echo "Unknown CUDA_VERSION ($CUDA_VERSION)" >&2; exit 1;;
  esac
  export TF_CUDNN_VERSION="7"
  export CUDA_TOOLKIT_PATH="/usr/local/cuda"
  export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"
  python3 ./configure.py
  bazel build --enable_runfiles build_pip_pkg
  bazel-bin/build_pip_pkg $STAGEDML_ROOT/_tfa
)}

installtfa() {(
  cd $STAGEDML_ROOT/_tfa
  sudo -H pip install --force `ls -t *whl | head -n 1`
)}

buildtf() {(
  set -e -x
  cd $STAGEDML_ROOT/3rdparty/tensorflow

  PYTHON_BIN_PATH="/usr/local/bin/python" \
  PYTHON_LIB_PATH="/usr/local/lib/python3.6/dist-packages" \
  TF_ENABLE_XLA="y" \
  TF_NEED_OPENCL_SYCL="n" \
  TF_NEED_ROCM="n" \
  TF_NEED_CUDA="y" \
  TF_DOWNLOAD_CLANG="n" \
  GCC_HOST_COMPILER_PATH="/usr/bin/gcc" \
  CC_OPT_FLAGS="-march=native -Wno-sign-compare" \
  TF_SET_ANDROID_WORKSPACE="n" \
  TF_NEED_TENSORRT=n \
  TF_CUDA_COMPUTE_CAPABILITIES="3.5,6.0" \
  TF_CUDA_CLANG=n \
  TF_NEED_MPI=n \
  ./configure
  bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package $STAGEDML_ROOT/_tf
  echo "Build finished. To install TF, type:"
  echo "     \`sudo -H pip3 install --force $STAGEDML_ROOT/_tf/<latest>.whl\`"
  echo "  or \`installtf\`"
)}

installtf() {(
  cd $STAGEDML_ROOT/_tf
  # Uninstall stock TF which comes from Docker image
  sudo -H pip uninstall -y tb-nightly tensorboard tensorflow \
                           tensorflow-estimator tensorflow-estimator-2.0-preview \
                           tf-nightly-gpu-2.0-preview
  # Install custom TF. See `buildtf`.
  sudo -H pip3 install --force `ls -t *whl | head -n 1` tensorflow-hub
)}

cat <<EOF
StagedML development environment assumes user to do the following:
- Clone StagedML repo in "\$STAGEDML_ROOT" ($STAGEDML_ROOT)
- Checkout git submodules of StagedML repo with
    \`git submodules update --init --recursive\`
- Build TensorFlow wheel from source with \`buildtf\`
- Install TensorFlow wheel with \`installtf\`
EOF
