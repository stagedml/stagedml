#!/bin/sh

apt-get install -y libssl1.0-dev nodejs-dev node-gyp npm
npm install -g vega-lite vega-cli canvas
pip3 install altair altair_saver altair_viewer vega
