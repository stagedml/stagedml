from os import (
    mkdir, makedirs, replace, listdir, rmdir, symlink, rename, remove, environ,
    walk, lstat, chmod, stat, readlink )
from os.path import join, basename, isfile, isdir
from textwrap import dedent
from contextlib import contextmanager
from numpy import load as np_load
from distutils.dir_util import copy_tree
from copy import deepcopy
from shutil import copyfile, copytree
from random import shuffle, random

