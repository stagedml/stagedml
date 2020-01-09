#!/bin/sh

BAZEL_VERSION="0.26.1"
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

{
echo "export BAZEL_VERSION=$BAZEL_VERSION"
} >>/etc/profile
