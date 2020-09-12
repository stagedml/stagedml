from os import (mkdir, makedirs, replace, listdir, rmdir, symlink, rename,
                remove, environ, walk, lstat, chmod, stat, readlink, cpu_count,
                getpid, get_terminal_size, getcwd, fsync )
from os.path import ( join, basename, isfile, isdir, islink, abspath )
from textwrap import dedent
from contextlib import contextmanager
from numpy import load as np_load
from distutils.dir_util import copy_tree
from copy import copy, deepcopy
from shutil import copyfile, copytree
from random import shuffle, random, Random, randint
from distutils.spawn import find_executable
from subprocess import ( Popen, run as os_run, PIPE, STDOUT )
from json import (loads as json_loads, load as json_load, dump as json_dump,
                  dumps as json_dumps )
from functools import partial
from itertools import chain
from bz2 import ( open as bz2_open )
from multiprocessing.pool import Pool
from beautifultable import BeautifulTable
from collections import defaultdict
from re import search as re_search
from hashlib import md5
from pygraphviz import AGraph
from pandas import DataFrame, read_csv
from collections import OrderedDict
from timeit import default_timer


from pylightnix import (Config, Stage, Manager, Context, Hash, Path, DRef, RRef,
                        RefPath, Closure, Build, BuildArgs, Matcher,
                        repl_realize, repl_continue, repl_build, build_outpath,
                        realize, rref2path, store_config, config_name,
                        mksymlink, isrref, isdir, dirhash, json_dump, json_load,
                        assert_serializable, assert_valid_rref, build_wrapper_,
                        readjson, store_rrefs, repl_rref, repl_cancel, rmref,
                        store_gc, instantiate, tryreadjson, tryreadjson_def,
                        mklens, Tag, RRefGroup, store_deps, store_initialize,
                        assert_store_initialized, build_setoutpaths,
                        build_cattrs, build_path, mkdrv, match_only,
                        match_latest, promise, mkconfig, build_wrapper,
                        build_wrapper_, store_cattrs, build_outpaths,
                        build_config, claim, fetchurl, fetchlocal, redefine,
                        dirsize, store_dref2path, path2rref, repl_cancelBuild,
                        repl_continueBuild, catref, lsref, shell, mknode,
                        repl_buildargs, repl_build, store_context, rref2dref)

def _stub_set_trace():
  print('set_trace(): ipdb module failed to load')

try:
  from ipdb import set_trace
except ModuleNotFoundError:
  set_trace = _stub_set_trace

