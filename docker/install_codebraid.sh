#!/bin/sh

PANDOC_VERSION=2.9.2
wget https://github.com/jgm/pandoc/releases/download/$PANDOC_VERSION/pandoc-${PANDOC_VERSION}-1-amd64.deb
dpkg -i pandoc-${PANDOC_VERSION}-1-amd64.deb
pip install codebraid
