# BERT fine-tuning experiments

``` python numberLines
import altair as alt
import pandas as pd
import numpy as np
from bert_finetune_experiment import *
from stagedml.stages.all import *
```

## Fine-tuning on GLUE tasks

``` python numberLines
experiment_allglue(n:int=1)->Dict[str,List[RRef]]:
  result={}
  for task_name in [t for t in glue_tasks() if t.upper() not in ['COLA']]:
    print(f"Fine-tuning {task_name}")
    batch_size={'MNLI-M':64,
                'MNLI-MM':64,
                'SNLI':64}.get(task_name.upper(),8)
    def _new_config(cfg:dict):
      cfg['train_batch_size']=batch_size
      cfg['train_epoches']=4
      return mkconfig(cfg)
    result[task_name]=realizeMany(instantiate(
      redefine(all_minibert_finetune_glue,
        new_config=_new_config, new_matcher=match_some(n)),
      task_name=task_name, num_instances=n))
  return result
```

``` python numberLines
results=experiment_allglue()
```

``` stdout
Fine-tuning SST-2
Fine-tuning MRPC
Fine-tuning QQP
Fine-tuning MNLI-m
Fine-tuning MNLI-mm
Fine-tuning SNLI
Fine-tuning QNLI
Fine-tuning RTE
Fine-tuning WNLI
```

``` python numberLines
t=BeautifulTable(max_width=1000)
t.set_style(BeautifulTable.STYLE_MARKDOWN)
t.width_exceed_policy = BeautifulTable.WEP_ELLIPSIS
t.column_headers=['Name']+list(results.keys())
t.append_row(['Accuracy']+[
  100*protocol_rref_metric(results[tn][0],'evaluate','eval_accuracy')
    for tn in results.keys()])
t.append_row(['F1_score']+[
  100*protocol_rref_metric(results[tn][0],'evaluate','f1_score')
    for tn in results.keys()])
t.append_row(['Training time, min']+[f"{store_buildelta(rrefs[0])/60:.1f}"
                                     for rrefs in list(results.values())])
print(t)
```

| Name               | SST-2  | MRPC   | QQP    | MNLI-m | MNLI-mm | SNLI   | QNLI   | RTE    | WNLI   |
| ------------------ | ------ | ------ | ------ | ------ | ------- | ------ | ------ | ------ | ------ |
| Accuracy           | 86.227 | 76.0   | 87.572 | 72.349 | 73.585  | 84.858 | 84.238 | 63.235 | 40.625 |
| F1\_score          | 55.0   | 75.784 | 43.775 | 43.673 | 44.235  | 40.232 | 53.395 | 38.915 | 33.696 |
| Training time, min | 13.3   | 0.9    | 70.0   | 34.8   | 34.8    | 48.3   | 20.3   | 0.7    | 0.4    |

Ref. [Upstream results](https://github.com/google-research/bert#bert)

## Batch size in BERT-\>MRPC fine-tuning

The top-level procedure of the experiment:

``` python numberLines
experiment_bs(n:int=1, exclude=[])->Dict[int,List[RRef]]:
  result={}
  for bs in [2,8,16,32,64]:
    def _new_config(cfg:dict):
      cfg['train_batch_size']=bs
      cfg['train_epoches']=5
      cfg['flags']=[f for f in cfg['flags'] if f not in exclude]
      return mkconfig(cfg)
    result[bs]=realizeMany(instantiate(
      redefine(all_minibert_finetune_glue, new_config=_new_config,
                                           new_matcher=match_some(n)),
      num_instances=n))
  return result


def experiment_trainmethod()->Dict[str,RRef]:
  result={}
  for tm in ['fit','custom']:
    def _new_config(cfg:dict):
      cfg['train_epoches']=5
      cfg['train_method']=tm
      return mkconfig(cfg)
    result[tm]=realize(instantiate(
      redefine(all_minibert_finetune_glue,
        new_config=_new_config, new_matcher=match_latest())))
  return result


def experiment_allglue(n:int=1)->Dict[str,List[RRef]]:
  result={}
  for task_name in [t for t in glue_tasks() if t.upper() not in ['COLA']]:
    print(f"Fine-tuning {task_name}")
    batch_size={'MNLI-M':64,
                'MNLI-MM':64,
                'SNLI':64}.get(task_name.upper(),8)
    def _new_config(cfg:dict):
      cfg['train_batch_size']=batch_size
      cfg['train_epoches']=4
      return mkconfig(cfg)
    result[task_name]=realizeMany(instantiate(
      redefine(all_minibert_finetune_glue,
        new_config=_new_config, new_matcher=match_some(n)),
      task_name=task_name, num_instances=n))
  return result
```

In the above code we fine-tune BERT-mini model on MRPC task. We increase
the number of epoch to 5 from the default 3, also we want to train
`num_instances=5` models of every batch\_size at once (Actually, current
implementation trains them one-by-one, we could parallelize this in
future). By using `match_some(5)` matcher we are saying that we want at
least 5 realizations of every configuration. Now we execute the
experiment.

``` python numberLines
results=experiment_bs(exclude=['+f1v2'])
```

Here is the code to process results. First, we build a collection of
`DataFrame`s.

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
    print(cols)
    dfs.append(DataFrame(cols))
ds=pd.concat(dfs)
```

``` stdout
{'steps': [3482, 6964, 10446, 13928, 17410], 'valid_accuracy': [0.7663043737411499, 0.7880434989929199, 0.7880434989929199, 0.8260869383811951, 0.804347813129425], 'batch_size': [2, 2, 2, 2, 2], 'iid': [0, 0, 0, 0, 0], 'valid_loss': [0.8980810046195984, 0.9873740077018738, 1.1126651763916016, 0.8929693698883057, 1.0717772245407104]}
{'steps': [3480, 6960, 10440, 13920, 17400], 'valid_accuracy': [0.75, 0.7663043737411499, 0.7771739363670349, 0.79347825050354, 0.8097826242446899], 'batch_size': [8, 8, 8, 8, 8], 'iid': [0, 0, 0, 0, 0], 'valid_loss': [0.5367669463157654, 0.5429021716117859, 0.5023452043533325, 0.553193986415863, 0.5063174962997437]}
{'steps': [3472, 6944, 10416, 13888, 17360], 'valid_accuracy': [0.7159090638160706, 0.7272727489471436, 0.7784090638160706, 0.75, 0.7556818127632141], 'batch_size': [16, 16, 16, 16, 16], 'iid': [0, 0, 0, 0, 0], 'valid_loss': [0.5679369568824768, 0.5936899781227112, 0.513241708278656, 0.5550832748413086, 0.5548913478851318]}
{'steps': [3456, 6912, 10368, 13824, 17280], 'valid_accuracy': [0.6625000238418579, 0.6937500238418579, 0.8062499761581421, 0.731249988079071, 0.7562500238418579], 'batch_size': [32, 32, 32, 32, 32], 'iid': [0, 0, 0, 0, 0], 'valid_loss': [0.5832815766334534, 0.5862361192703247, 0.496770441532135, 0.5832360982894897, 0.533299446105957]}
{'steps': [3456, 6912, 10368, 13824, 17280], 'valid_accuracy': [0.65625, 0.6953125, 0.7109375, 0.7265625, 0.6875], 'batch_size': [64, 64, 64, 64, 64], 'iid': [0, 0, 0, 0, 0], 'valid_loss': [0.6326533555984497, 0.5831466317176819, 0.5478577613830566, 0.6125747561454773, 0.6100413203239441]}
```

Finally, we print the validation accuracy of our models. Here:

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
