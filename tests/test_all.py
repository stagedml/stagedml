from stagedml.imports.sys import (instantiate, realize, repl_realize,
                                  repl_continueBuild, repl_cancelBuild, isrref,
                                  repl_buildargs, isrref, build_setoutpaths,
                                  shell)

from stagedml.stages.all import (all_fetch_glue, all_tfrec_glue,
                                 all_fetch_bertcp,
                                 bert_finetune_glue_zhg)

from stagedml.stages.bert_finetune_glue_zhg import *

def all_test_bert(m):
  ckpt = all_fetch_bertcp(m)
  glue = all_tfrec_glue(m, task_name='MRPC', lower_case=True)
  bert = bert_finetune_glue_zhg(m, ckpt, glue)
  return bert

def debug_bert()->State:
  s=State(repl_buildargs(repl_realize(instantiate(all_test_bert))))
  build_setoutpaths(s,1)
  return s

def test_bert():
  rref = realize(instantiate(all_test_bert))
  assert isrref(rref)
