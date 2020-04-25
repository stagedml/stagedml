Test summary on BERT pre-training
---------------------------------

TODO: intro

``` {.python .numberLines startFrom="1"}
import altair as alt
import pandas as pd
import numpy as np
from bert_pretrain_experiment import *
from stagedml.stages.all import *
```

TODO: Try to examine the source code of the function

We now run the experiment. We assume that Pylightnix already has the
results in it's storage because otherwize the report generation would
take too long to complete.

``` {.python .numberLines startFrom="6"}
pretrained,finetuned=run() # TODO: assert cached
```

``` {.stdout}
Begin pretraining
Pre-training up to epoch 20
Fine-tunining up to epoch 20
Pre-training up to epoch 40
Fine-tunining up to epoch 40
Pre-training up to epoch 60
Fine-tunining up to epoch 60
Pre-training up to epoch 80
Fine-tunining up to epoch 80
Pre-training up to epoch 100
Fine-tunining up to epoch 100
Pre-training up to epoch 120
Fine-tunining up to epoch 120
Pre-training up to epoch 140
Fine-tunining up to epoch 140
Pre-training up to epoch 160
Fine-tunining up to epoch 160
Pre-training up to epoch 180
Fine-tunining up to epoch 180
Pre-training up to epoch 200
Fine-tunining up to epoch 200
```

TODO: results

``` {.python .numberLines startFrom="7"}

results=defaultdict(list)
for epoch,rref in finetuned.items():
  for subf,metric in zip(['validation','train'],
                         ['epoch_accuracy','batch_accuracy']):
    es=tensorboard_scalar_events(
        rref,subf,metric)
    results[metric].append(
      pd.DataFrame({'step':[e.step for e in es],
                    'value':[e.value for e in es],
                    'pretrained':[epoch for _ in es]}))
```

TODO: Describe train batch accuracy

``` {.python .numberLines startFrom="18"}
metric='batch_accuracy'
dflist=results[metric]
df=pd.concat(dflist)
chart=alt.Chart(df).mark_line().encode(
  x='step', y='value', color='pretrained')
altair_print(chart, f'figure_{metric}.png')
```

![](./figure_batch_accuracy.png)

TODO: Describe evaluation epoch accuracy

``` {.python .numberLines startFrom="24"}
metric='epoch_accuracy'
dflist=results[metric]
df=pd.concat(dflist)
chart=alt.Chart(df).mark_line().encode(
  x='step', y='value', color='pretrained')
altair_print(chart, f'figure_{metric}.png')
```

![](./figure_epoch_accuracy.png)
