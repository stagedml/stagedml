from pylightnix import *
from pygraphviz import AGraph
from stagedml.stages.all import *
from stagedml.core import depgraph

STAGES:List[Stage]=[
    all_convnn_mnist,
    all_transformer_nl2bash,
    all_transformer_wmtenru,
    partial(all_bert_finetune_glue,task_name='MRPC'),
    all_minibert_pretrain,
    ]


def mkgraph(stages:List[Stage]=STAGES, filename:Optional[str]=None,
    layout:str='dot')->AGraph:
  """ Build a graph of dependencies for given stages. If `filename` is not
  None, save the graph into this file. """
  G=AGraph(strict=False,directed=True)
  touched:Set[DRef]=set()
  frontier=[instantiate(s).dref for s in stages]
  while len(frontier)>0:
    dref=frontier.pop()
    G.add_node(mklens(dref).name.val)
    for dep_dref in store_deps([dref]):
      G.add_node(mklens(dep_dref).name.val)
      G.add_edge(mklens(dref).name.val, mklens(dep_dref).name.val or dep_dref)
      if dep_dref not in touched:
        frontier.append(dep_dref)
      touched.add(dep_dref)

  if layout is not None:
    G.layout(prog=layout)
  if filename is not None:
    G.draw(filename)
  return G

def mkgraph_finetune()->None:
  G=depgraph(stages=[partial(all_minibert_finetune_glue, task_name=t) for t
    in glue_tasks()])
  G.layout(prog='dot')
  G.draw(f'graph-finetune.png')

def mkgraph_pretrain()->None:
  depgraph(stages=[all_minibert_pretrain],
    filename=f'graph-pretrain.png', layout='dot')


def mkgraph_demo()->None:
  """ Build a graph demonstrated in top-level README.md. We change some
  misleading names before processing """

  def _pretrain_stage(nepoch:int)->Stage:
    return partial(all_minibert_pretrain,train_epoches=nepoch)

  def _finetune_stage(task_name:str, nepoch:int)->Stage:
    def _stage(m)->BertGlue:
      refglue=all_fetchglue(m)
      refbert=_pretrain_stage(nepoch)(m)
      gluetfr=glue_tfrecords(m, task_name,
          bert_vocab=mklens(refbert).bert_vocab.refpath,
          lower_case=mklens(refbert).cased.val==False,
          refdataset=refglue)
      def _nc(c):
        mklens(c).name.val='mini'+c['name']
      tfbert=redefine(bert_finetune_glue,new_config=_nc)(m,refbert,gluetfr)
      return tfbert
    return _stage

  depgraph(stages=[_finetune_stage(t,1000) for t in ['MRPC', 'MNLI-m', 'SST-2']],
    filename=f'graph-demo.png', layout='dot')

