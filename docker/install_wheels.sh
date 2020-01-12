#!/bin/sh

cat >>/etc/bash.bashrc <<EOF
(
set -e -x
cd /workspace/3rdparty/pylightnix
rm -rf build dist || true
python3 setup.py sdist bdist_wheel
sudo pip3 install --force dist/*whl
)

(
set -e -x
cd /workspace
rm -rf build dist || true
python3 setup.py sdist bdist_wheel
sudo pip3 install --force dist/*whl
)
EOF
