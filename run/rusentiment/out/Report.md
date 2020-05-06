RuSentiment
===========

TODO: intro

``` {.python .numberLines startFrom="1"}
import altair as alt
import pandas as pd
import numpy as np
from rusentiment_experiment import *
from stagedml.stages.all import *
```

Model runner code

``` {.html .numberLines startFrom="1"}
class Runner:
  def __init__(self, rref:RRef):
    self.rref=rref
    self.tokenizer=FullTokenizer(
      vocab_file=mklens(rref).tfrecs.bert_vocab.syspath,
      do_lower_case=mklens(rref).tfrecs.lower_case.val)
    self.max_seq_length=mklens(rref).max_seq_length.val
    self.model=BertClsModel(
      BuildArgs(rref2dref(rref), store_context(rref), None, {}, {}))
    bert_finetune_build(self.model)
    self.model.model.load_weights(mklens(rref).checkpoint_full.syspath)

  def eval(self, sentences:List[str], batch_size:int=10)->List[List[float]]:

    @tf.function
    def _tf_step(inputs):
      return self.model.model_eval(inputs, training=False)

    def _step(inputs:List[Tuple[Any,Any,Any]]):
      return _tf_step((tf.constant([i[0] for i in inputs], dtype=tf.int64),
                       tf.constant([i[1] for i in inputs], dtype=tf.int64),
                       tf.constant([i[2] for i in inputs], dtype=tf.int64)))\
             .numpy().tolist()

    buf=[]
    outs:List[List[float]]=[]
    for i,s in enumerate(sentences):
      ie=InputExample(guid=f"eval_{i:04d}", text_a=s, label='dummy')
      f=convert_single_example(
        10, ie, ['dummy'], self.max_seq_length, self.tokenizer)
      buf.append((f.input_ids, f.input_mask, f.segment_ids))
      if len(buf)>=batch_size:
        outs.extend(_step(buf))
        buf=[]

    outs.extend(_step(buf))
    return outs

# runner ends
```

Evaluation and post-processing code

``` {.html .numberLines startFrom="1"}
def bert_rusentiment_evaluation(m:Manager)->DRef:

  def _realize(b:Build):
    build_setoutpaths(b,1)
    r=Runner(mklens(b).model.rref)
    df=read_csv(mklens(b).model.tfrecs.refdataset.output_tests.syspath)
    labels=sorted(df['label'].value_counts().keys())
    df['pred']=[labels[np.argmax(probs)] for probs in r.eval(list(df['text']))]
    confusion={l:{l2:0.0 for l2 in labels} for l in labels}
    for i,row in df.iterrows():
      confusion[row['label']][row['pred']]+=1.0
    confusion={l:{l2:i/sum(items.values()) for l2,i in items.items()} \
               for l,items in confusion.items()}
    with open(mklens(b).confusion_matrix.syspath,'w') as f:
      json_dump(confusion,f,indent=4)
    df.to_csv(mklens(b).prediction.syspath)

  return mkdrv(m, matcher=match_only(), realizer=build_wrapper(_realize),
    config=mkconfig({
      'model':all_multibert_finetune_rusentiment(m),
      'confusion_matrix':[promise, 'confusion_matrix.json'],
      'prediction':[promise, 'prediction.csv'],
      'version':2,
      }))

# eval ends
```

Print the confusion matrix

``` {.python .numberLines startFrom="6"}
rref=realize(instantiate(bert_rusentiment_evaluation))
data:dict={'label':[],'pred':[],'val':[]}
for l,items in readjson(mklens(rref).confusion_matrix.syspath).items():
  for l2,val in items.items():
    data['label'].append(l)
    data['pred'].append(l2)
    data['val'].append(f"{val:.2f}")

base=alt.Chart(DataFrame(data))
r=base.mark_rect().encode(
  y='label:O',x='pred:O',color='val:Q')
t=base.mark_text().encode(y='label:O',x='pred:O',text='val:Q')
altair_print((r+t).properties(width=300,height=300), 'cm.png')
```

![](./cm.png)
