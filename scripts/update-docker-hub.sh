#!/bin/sh
# Bootstrap the creation of 'User' docker image and upload it to the DockerHub.

set -e -x

./rundocker.sh -- make wheel
./rundocker.sh docker/stagedml_user.docker -- ls /
docker login
docker push stagedml/user:latest
