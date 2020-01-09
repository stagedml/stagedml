#!/bin/sh

# Run chromium as a separate user and with strange ports enabled
# Needed to access Tensorboard or Jupyter

mkdir -p $HOME/.chrome_mrc-nlp || true
chromium \
  --user-data-dir=$HOME/.chrome_mrc-nlp \
  --explicitly-allowed-ports=`seq -s , 6000 1 6100`,`seq -s , 8000 1 8100`,7777 \
  http://127.0.0.1:`expr 6000 + $UID - 1000` "$@"
