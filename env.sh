if test -z "$STAGEDML_ROOT" ; then
  export STAGEDML_ROOT=`pwd`
fi

export CWD="$STAGEDML_ROOT"
export TERM=xterm-256color
export PATH="$STAGEDML_ROOT/.nix_docker_inject.env/bin:$PATH"
export PYTHONPATH=""
export MYPYPATH=""
for p in \
  $STAGEDML_ROOT/3rdparty/tensorflow-models \
  $STAGEDML_ROOT/3rdparty/modelcap \
  $STAGEDML_ROOT/ \
  ; do
  if test -d "$p" ; then
    export PYTHONPATH="$p:$PYTHONPATH"
    export MYPYPATH="$p:$MYPYPATH"
    echo "Directory '$p' added to PYTHONPATH" >&2
  else
    echo "Directory '$p' doesn't exists. Not adding to PYTHONPATH" >&2
  fi
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

cudarestart() {
  sudo rmmod nvidia_uvm ; sudo modprobe nvidia_uvm
}

runnetron() {
  netron --host 0.0.0.0 -p 6006 "$@"
}

conftf() {(
  set -e -x
  cd $STAGEDML_ROOT/3rdparty/tensorflow
  export TF_NCCL_VERSION="$NCCL_VERSION"   # <-- Required for TF2.0 + CUDA
  ./configure
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
  rm -rf $STAGEDML_ROOT/_tf 2>/dev/null || true
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package $STAGEDML_ROOT/_tf
  echo "Build finished. To install TF, type:"
  echo "     \`sudo -H pip3 install --force $STAGEDML_ROOT/_tf/*whl\`"
  echo "  or \`installtf\`"
)}

installtf() {
  # Uninstall stock TF which comes from Docker image
  sudo -H pip uninstall -y tb-nightly tensorboard tensorflow \
                           tensorflow-estimator tensorflow-estimator-2.0-preview \
                           tf-nightly-gpu-2.0-preview
  # Install custom TF2.1. See `conftf`/`buildtf`.
  sudo -H pip3 install --force $STAGEDML_ROOT/_tf/*whl tensorflow-hub
}


