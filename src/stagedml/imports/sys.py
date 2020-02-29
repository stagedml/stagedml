from os import environ
from os.path import join, basename, isfile, isdir
from textwrap import dedent
from contextlib import contextmanager
from numpy import load as np_load
from distutils.dir_util import copy_tree
from copy import deepcopy
