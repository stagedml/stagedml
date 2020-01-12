# Tell login shell to source `./env.sh` at start. This will set PYTHONPATH and
# add development helper functions

{
echo "export STAGEDML_ROOT=\$HOME"
echo "if test -f \"\$STAGEDML_ROOT/env.sh\" ; then"
echo "  . \$STAGEDML_ROOT/env.sh"
echo "fi"
} >>/etc/bash.bashrc

