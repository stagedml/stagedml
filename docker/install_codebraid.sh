#!/bin/sh

PANDOC_VERSION=2.7
wget https://github.com/jgm/pandoc/releases/download/$PANDOC_VERSION/pandoc-${PANDOC_VERSION}-1-amd64.deb
dpkg -i pandoc-${PANDOC_VERSION}-1-amd64.deb
pip3 install git+https://github.com/gpoore/codebraid@v0.4.0
