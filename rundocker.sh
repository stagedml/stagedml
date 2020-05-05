#!/bin/sh

CWD=$(cd `dirname $0`; pwd;)

GITHACK=n
MAPPORTS=`expr $UID - 1000`
DOCKER_FILENAME=$CWD/docker/stagedml_dev.docker
DOCKER_IMGNAME=""

while test -n "$1" ; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [-n|--no-map-ports] " \
           "[-m INT|--map-ports-base INT] " \
           "(FILE_NAME.docker|DOCKER_IMAGE_NAME)" >&2
      exit 1
      ;;
    -n|--no-map-ports)
      MAPPORTS=""
      ;;
    -m|--map-ports-base)
      MAPPORTS="$2"; shift
      ;;
    *)
      if test -f "$1" ; then
        DOCKER_FILENAME="$1"
      else
        DOCKER_IMGNAME="$1"
      fi
      ;;
  esac
  shift
done

# The code snippet below builds and injects nix packages into the Docker
# container, if the specified expression file does exist. It is a matter of
# comfort and is not actually required for system operation.
if which nix-build && test -f "$CWD/nix/docker_inject.nix" ; then
  nix-build $CWD/nix/docker_inject.nix --argstr me $USER --out-link "$CWD/.nix_docker_inject.env"
  if test "$?" != "0" ; then
    echo "nix-build failed, continue without nix injections" >&2
  fi
else
  echo "No nix-expressions to inject into this container" >&2
fi

# Remap detach key to Ctrl+e,e
DOCKER_CFG="/tmp/docker-stagedml-$UID"
mkdir "$DOCKER_CFG" 2>/dev/null || true
cat >$DOCKER_CFG/config.json <<EOF
{ "detachKeys": "ctrl-e,e" }
EOF

set -e -x

if test -n "$DOCKER_IMGNAME" ; then
  docker pull "$DOCKER_IMGNAME:latest"
else
  DOCKER_SUFFIX=`echo $DOCKER_FILENAME | sed -n 's@.*_\(.*\)\.docker$@\1@p'`
  DOCKER_IMGNAME=stagedml/$DOCKER_SUFFIX

  docker build \
    --build-arg=http_proxy=$https_proxy \
    --build-arg=https_proxy=$https_proxy \
    --build-arg=ftp_proxy=$https_proxy \
    -t "$DOCKER_IMGNAME" \
    -f "$DOCKER_FILENAME" "$CWD/docker"
fi

if test -n "$MAPPORTS"; then
  PORT_TENSORBOARD=`expr 6000 + $MAPPORTS`
  PORT_JUPYTER=`expr 8000 + $MAPPORTS`

  DOCKER_PORT_ARGS="
    -p 0.0.0.0:$PORT_TENSORBOARD:6006 -e PORT_TENSORBOARD=$PORT_TENSORBOARD
    -p 0.0.0.0:$PORT_JUPYTER:8888 -e PORT_JUPYTER=$PORT_JUPYTER"
fi

# To allow X11 connections from docker
xhost +local: || true
cp "$HOME/.Xauthority" "$CWD/.Xauthority" || true

if which nvidia-docker >/dev/null 2>&1; then
  DOCKER_CMD=nvidia-docker
else
  DOCKER_CMD=docker
fi

# Mount additional folders inside the container
if test -d "/home/data" ; then
  DOCKER_MOUNT_ARGS="$DOCKER_MOUNT_ARGS -v /home/data:/home/data"
fi
if test -d "/nix" ; then
  DOCKER_MOUNT_ARGS="$DOCKER_MOUNT_ARGS -v /nix:/nix"
fi
if test -d "/tmp/.X11-unix" ; then
  DOCKER_MOUNT_ARGS="$DOCKER_MOUNT_ARGS -v /tmp/.X11-unix:/tmp/.X11-unix"
fi
if test -d "/dev/bus/usb" ; then
  DOCKER_MOUNT_ARGS="$DOCKER_MOUNT_ARGS --privileged -v /dev/bus/usb:/dev/bus/usb"
fi

${DOCKER_CMD} --config "$DOCKER_CFG" \
    run -it --rm \
    --volume "$CWD:/workspace" \
    --workdir /workspace \
    -m 32g \
    -e HOST_PERMS="$(id -u):$(id -g)" \
    -e "CI_BUILD_HOME=/workspace" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    -e "DISPLAY=$DISPLAY" \
    -e "EDITOR=$EDITOR" \
    -e "TERM=$TERM" \
    ${DOCKER_PORT_ARGS} \
    ${DOCKER_MOUNT_ARGS} \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --privileged -v /dev/bus/usb:/dev/bus/usb \
    "${DOCKER_IMGNAME}" \
    bash /install/with_the_same_user.sh bash --login

