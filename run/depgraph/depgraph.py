from stagedml.stages.all import *
from pylightnix import *
from pygraphviz import AGraph

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
  for mode in ['major','KK','hier','ipsep','maxent']:
    for l in ['dot']:
      G=mkgraph(stages=[partial(all_minibert_finetune_glue, task_name=t) for t
        in glue_tasks()])
      # G.graph_attr['size']='1000'
      # G.graph_attr['mode']=mode
      # G.graph_attr['levelsgap']=2
      # G.graph_attr['newrank']=True
      # G.graph_attr['pagedir']='LT'
      # G.graph_attr['packmode']='graph'
      # G.graph_attr['pencolor']='red:black'
      G.layout(prog=l)
      G.draw(f'graph-finetune-{l}-{mode}.png')

def mkgraph_pretrain()->None:
  for l in ['neato','dot','twopi','circo','fdp']:
    mkgraph(stages=[all_minibert_pretrain],
      filename=f'graph-pretrain-{l}.png', layout=l)


def mkgraph_demo()->None:

  def _pretrain_stage(nepoch:int)->Stage:
    return partial(all_minibert_pretrain,train_epoches=nepoch)

  def _finetune_stage(task_name:str, nepoch:int)->Stage:
    def _stage(m)->BertGlue:
      refglue=all_fetchglue(m)
      refbert=_pretrain_stage(nepoch)(m)
      gluetfr=glue_tfrecords(m, task_name,
          bert_vocab=mklens(refbert).bert_vocab.refpath,
          refdataset=refglue)
      def _nc(cfg):
        cfg['name']='mini'+cfg['name']
        return mkconfig(cfg)
      tfbert=redefine(bert_finetune_glue,new_config=_nc)(m,refbert,gluetfr)
      return tfbert
    return _stage

  mkgraph(stages=[_finetune_stage(t,1000) for t in ['MRPC', 'MNLI-m', 'SST-2']],
    filename=f'graph-demo.png', layout='dot')

