#!/bin/sh
# Run Chromium browser in attempt to visit local TensorBoard or Jupyter pages.
# Chromium is run with a custom profile, and with a number of ports unprotected.

mkdir -p $HOME/.chrome_stagedml_profile || true
chromium \
  --user-data-dir=$HOME/.chrome_stagedml_profile \
  --explicitly-allowed-ports=`seq -s , 6000 1 6100`,`seq -s , 8000 1 8100`,7777 \
  http://127.0.0.1:`expr 6000 + $UID - 1000` "$@"
