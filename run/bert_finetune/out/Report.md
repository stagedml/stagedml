# BERT fine-tuning experiments

``` python numberLines
import altair as alt
import pandas as pd
import numpy as np
from bert_finetune_experiment import *
from stagedml.stages.all import *
from stagedml.core import depgraph
```

## Fine-tuning on GLUE tasks

``` python numberLines
experiment_allglue(n:int=1)->Dict[str,List[RRef]]:
  result_allglue={}
  for task_name in [t for t in glue_tasks() if t.upper() not in ['COLA']]:
    print(f"Fine-tuning {task_name}")
    batch_size={'MNLI-M':64,
                'MNLI-MM':64,
                'SNLI':64}.get(task_name.upper(),8)
    def _new_config(cfg:dict):
      cfg['train_batch_size']=batch_size
      cfg['train_epoches']=4
      return mkconfig(cfg)
    result_allglue[task_name]=realizeMany(instantiate(
      redefine(all_minibert_finetune_glue,
        new_config=_new_config, new_matcher=match_some(n)),
      task_name=task_name, num_instances=n))
  return result_allglue
```

Here, we:

  - Loop over all tasks excluding COLA (TODO: remember why do we exclude
    COLA?)
  - For every task, we realize minibert model
      - `realizeMany(instantiate(...))` is the generic procedure of
        realizing Pylightnix stages
      - `redefine(..)` allows us re-define stage’s configuration
        in-place. In our case we adjust parameters to match the upstream
        settings (set `batch_size`, 4 epoch)
      - Note, that we don’t evaluate all possible parameters like the
        upstream did due to time/hardware constraints.
      - `all_minibert_finetune_glue` is one of StagedML stages, defined
        in `stagedml.stages.all`. By realizing it we also realize all
        it’s dependencies, which includes fetching the required images
        and datasets from the Internet.
  - We build a dictionary containing `RRef` realization reference for
    every task.

Below we display the graph of stages to be realized in order to realize
the top-level stage of `all_minibert_finetune_glue(task_name='QQP')`:

``` python numberLines
p=join(environ['REPOUT'],"graph.png")
depgraph([partial(all_minibert_finetune_glue,task_name='QQP')],filename=p)
print(f"![]({p})")
```

![](out/graph.png)

``` python numberLines
results=experiment_allglue()
```

We display results in a table

``` python numberLines
t=BeautifulTable(max_width=1000)
t.set_style(BeautifulTable.STYLE_MARKDOWN)
t.width_exceed_policy = BeautifulTable.WEP_ELLIPSIS
t.column_headers=['Name']+list(results.keys())
t.append_row(['Accuracy, %']+[
  100*protocol_rref_metric(results[tn][0],'evaluate','eval_accuracy')
    for tn in results.keys()])
t.append_row(['F1_score*100']+[
  100*protocol_rref_metric(results[tn][0],'evaluate','f1_score')
    for tn in results.keys()])
t.append_row(['Tr.time, min']+[f"{store_buildelta(rrefs[0])/60:.1f}"
                                  for rrefs in list(results.values())])
print(t)
```

| Name           | SST-2  | MRPC   | QQP    | MNLI-m | MNLI-mm | SNLI   | QNLI   | RTE    | WNLI   |
| -------------- | ------ | ------ | ------ | ------ | ------- | ------ | ------ | ------ | ------ |
| Accuracy, %    | 86.227 | 76.0   | 87.572 | 72.349 | 73.585  | 84.858 | 84.238 | 63.235 | 40.625 |
| F1\_score\*100 | 55.0   | 75.784 | 43.775 | 43.673 | 44.235  | 40.232 | 53.395 | 38.915 | 33.696 |
| Tr.time, min   | 13.3   | 0.9    | 70.0   | 34.8   | 34.8    | 48.3   | 20.3   | 0.7    | 0.4    |

Ref. [Upstream results](https://github.com/google-research/bert#bert)

## Batch size in BERT-\>MRPC fine-tuning

The top-level procedure of the experiment:

``` python numberLines
experiment_bs(n:int=1, exclude=[])->Dict[int,List[RRef]]:
  result_bs={}
  for bs in [2,8,16,32,64]:
    def _new_config(cfg:dict):
      cfg['train_batch_size']=bs
      cfg['train_epoches']=5
      cfg['flags']=[f for f in cfg['flags'] if f not in exclude]
      return mkconfig(cfg)
    result_bs[bs]=realizeMany(instantiate(
      redefine(all_minibert_finetune_glue, new_config=_new_config,
                                           new_matcher=match_some(n)),
      num_instances=n))
  return result_bs
```

In the above code we:

  - Loop over certain batch\_sizes. For every batch\_size we evaluate
    min-bert model
      - `realizeMany(instance(..))` runs generice two-pass stage
        realization mechanism of Pylightnix.
      - `redefine(..)` tweaks stage configuration before the
        realization. Besides setting batch\_size, we increase number of
        epoches up to 5.
      - `all_minibert_finetune_glue` is one of StagedML stages, defined
        in `stagedml.stages.all`. By realizing it we also realize all
        it’s dependencies, which includes fetching the required images
        and datasets from the Internet.
  - In a version of this experiment we could increase a number stages
    instances which shares same configuration by setting
    `num_instances`.
  - We collect resulting realization references in a dictionary.

<!-- end list -->

``` python numberLines
results=experiment_bs(exclude=['+f1v2'])
```

Results are shown below.

``` python numberLines
dfs=[]
for bs,rrefs in results.items():
  for iid,rref in enumerate(rrefs):
    cols={}
    es=tensorboard_tensor_events(rref,'valid','accuracy')
    assert len(es)>0
    cols['steps']=[e.step*bs for e in es]
    cols['valid_accuracy']=[te2float(e) for e in es]
    cols['batch_size']=[bs for _ in es]
    cols['iid']=[iid for _ in es]
    es=tensorboard_tensor_events(rref,'valid','loss')
    assert len(cols['steps'])==len(es)
    cols['valid_loss']=[te2float(e) for e in es]
    dfs.append(DataFrame(cols))
ds=pd.concat(dfs)
```

Comments:

  - `tensorboard_tensor_events` is a helper method which access stages
    tensorboard journals stored in realization folder.
  - `iid` is the instance identifier of the model.
  - `batch_size` is the batch size used during fine-tuning
  - `steps` is the number of sentences passed through the model.
    According to the value of `max_seq_length` parameter, each sentence
    contains maximum 128 tokens.

<!-- end list -->

``` python numberLines
metric='valid_accuracy'
chart=alt.Chart(ds).mark_line().encode(
  x='steps', y=metric, color='batch_size',
  strokeDash='iid')
altair_print(chart, f'figure_{metric}.png')
```

![](./figure_valid_accuracy.png)

``` stderr
WARN StrokeDash channel should be used with only discrete data.
WARN Using discrete channel "strokeDash" to encode "quantitative" field can be misleading as it does not encode magnitude.
```

  - TODO: find out why do models with smaller batch sizes train better?
      - Is it the effect of batch-normalization?
      - Is it the effect of un-disabled dropout?

## Junk

    {.python .cb.nb show=code+stdout:raw+stderr}
    metric='train-lr'
    dflist=results[metric]
    df=pd.concat(dflist)
    df=df[(df['batch_size']==32) | (df['batch_size']==2)]
    chart=alt.Chart(df).mark_line().encode(
      x='step', y='value', color='batch_size',
      strokeDash='optver')
    altair_print(chart, f'figure_{metric}.png')
