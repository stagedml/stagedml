from stagedml.imports.sys import (instantiate, realize, repl_realize,
                                  repl_continueBuild, repl_cancelBuild, isrref,
                                  repl_buildargs, isrref, build_setoutpaths,
                                  shell)

from stagedml.stages.all import (all_fetch_glue, all_tfrec_glue,
                                 all_fetch_bertcp, bert_finetune_glue_zhg,
                                 all_tfrec_rusent, all_fetch_rusent)

from stagedml.stages.bert_finetune_glue_zhg import *

def all_test_bert_glue(m):
  bertcp = all_fetch_bertcp(m)
  glue = all_tfrec_glue(m, task_name='MRPC', lower_case=True, bertcp=bertcp)
  bert = bert_finetune_glue_zhg(m, bertcp, glue)
  return bert

def all_test_bert_rusent(m):
  bertcp = all_fetch_bertcp(m)
  glue = all_tfrec_rusent(m, bertcp=bertcp)
  bert = bert_finetune_glue_zhg(m, bertcp, glue)
  return bert

def debug_bert_glue()->State:
  s=State(repl_buildargs(repl_realize(instantiate(all_test_bert_glue))))
  build_setoutpaths(s,1)
  return s

def check_bert_glue():
  rref = realize(instantiate(all_test_bert_glue))
  assert isrref(rref)
  return rref
