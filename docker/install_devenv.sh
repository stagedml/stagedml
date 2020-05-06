# Tell login shell to source `./env.sh` at start. This will set PYTHONPATH and
# add development helper functions. Much of functionality depends on the fact
# that `STAGEDML_ROOT` contains the correct path to StagedML repository.

{
echo "if test -f /install/devenv.sh ; then"
echo "  . /install/devenv.sh"
echo "fi"
} >>/etc/bash.bashrc

