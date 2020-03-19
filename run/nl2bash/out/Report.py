
import numpy as np
import matplotlib.pyplot as plt

from shutil import copyfile
from itertools import islice
from pylightnix import (
    RRef, Path, realize, realizeMany, instantiate, redefine, mkconfig, promise,
    rref2dref, mksymlink, rref2path, mklens, match_best, match_some )
from stagedml.imports import ( environ, join, environ, makedirs )
from stagedml.stages.all import ( transformer_wmt, all_nl2bashsubtok,
    all_fetchnl2bash )
from analyze import ( read_tensorflow_log, vocab_size, model_size )

def baseline_subtok(m):
  return all_nl2bashsubtok(m, shuffle=True,
                              with_bash_charset=False,
                              with_bash_subtokens=False)

def baseline_transformer(m):
  def _config(c):
    c['train_steps']=6*5000
    c['params']['beam_size']=3 # As in Tellina paper
    return mkconfig(c)
  return redefine(transformer_wmt,
                  new_config=_config,
                  new_matcher=match_some())(m, baseline_subtok(m), num_instances=5)

rref=realize(instantiate(all_fetchnl2bash))
copyfile(mklens(rref).eval_input_combined.syspath, join(environ['REPORT_OUTPATH'],'eval_input.txt'))
copyfile(mklens(rref).eval_target_combined.syspath, join(environ['REPORT_OUTPATH'],'eval_target.txt'))

with open(mklens(rref).train_input_combined.syspath) as inp, \
     open(mklens(rref).train_target_combined.syspath) as tgt:
  for i, (iline, tline) in islice(enumerate(zip(inp,tgt)),5):
    print(f"#{i}\t[I] {iline.strip()}\n\t[T] {tline.strip()}")

plt.figure(1)
plt.xlabel("Epoches")
plt.title("BLEU-cased, Baseline transformer")

out=Path(join(environ['STAGEDML_ROOT'],'_experiments','nl2bash','baseline'))
makedirs(out, exist_ok=True)
summary_baseline_bleu=[]
for i,rref in enumerate(realizeMany(instantiate(baseline_transformer))):
  mksymlink(rref, out, f'run-{i}', withtime=False)
  baseline_bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
  plt.plot(range(len(baseline_bleu)), baseline_bleu, label=f'run-{i}', color='blue')
  summary_baseline_bleu.append((vocab_size(baseline_transformer),baseline_bleu[4]))

plt.legend(loc='upper left', frameon=True)
plt.grid(True)

rref=realize(instantiate(redefine(baseline_transformer,new_matcher=match_best('bleu.txt'))))
baseline_bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')

rref=realize(instantiate(redefine(baseline_transformer,new_matcher=match_best('bleu.txt'))))
copyfile(join(rref2path(rref),'output-5.txt'), join(environ['REPORT_OUTPATH'],'baseline_output.txt'))

def unshuffled_subtok(m):
  return all_nl2bashsubtok(m, shuffle=False,
                              with_bash_charset=False,
                              with_bash_subtokens=False)

def unshuffled_transformer(m):
  def _config(c):
    c['train_steps']=6*5000
    c['params']['beam_size']=3 # As in Tellina paper
    return mkconfig(c)
  return redefine(transformer_wmt,_config)(m, unshuffled_subtok(m))

plt.figure(2)
plt.xlabel("Epoches")
plt.title("BLEU, Unshuffled transformer")

out=Path(join(environ['STAGEDML_ROOT'],'_experiments','nl2bash','unshuffled'))
makedirs(out, exist_ok=True)
rref=realize(instantiate(unshuffled_transformer))
mksymlink(rref, out, 'result', withtime=False)
unshuffled_bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
plt.plot(range(len(unshuffled_bleu)), unshuffled_bleu, label=f'run', color='red')
summary_unshuffled_bleu=[(vocab_size(unshuffled_transformer),unshuffled_bleu[4])]

plt.legend(loc='upper left', frameon=True)
plt.grid(True)

def run1(vsize:int):

  def mysubtok(m):
    def _config(d):
      d['target_vocab_size']=vsize  # Doesn't in fact depend on this parameter
      d['vocab_file'] = [promise, 'vocab.%d' % vsize]
      return mkconfig(d)
    return redefine(all_nl2bashsubtok, _config)(m,
                    shuffle=True, with_bash_charset=True, with_bash_subtokens=True)

  def mytransformer(m):
    def _config(c):
      c['train_steps']=5*5000
      c['params']['beam_size']=3 # As in Tellina paper
      return mkconfig(c)
    return redefine(transformer_wmt,_config)(m, mysubtok(m))

  return mysubtok, mytransformer

plt.figure(2)
plt.xlabel("Epoches")
plt.title("BLEU, Bash-specific tokens")

plt.plot(range(len(unshuffled_bleu)), unshuffled_bleu, label=f'Unshuffled transformer', color='red')
plt.plot(range(len(baseline_bleu)), baseline_bleu, label=f'Baseline transformer', color='orange')

out=Path(join(environ['STAGEDML_ROOT'],'_experiments','nl2bash','bashspec'))
makedirs(out, exist_ok=True)
summary_bashtoken_bleu=[]
for i,vsize in enumerate([ 30000, 25000, 20000, 15000, 10000, 5000, 1000, 500 ]) :
  mysubtok,mytransformer=run1(vsize)
  rref=realize(instantiate(mytransformer))
  mksymlink(rref, out, f'run-{vocab_size(mytransformer)}', withtime=False)
  bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
  plt.plot(range(len(bleu)), bleu, label=f'run-{vocab_size(mytransformer)}', color='blue')
  summary_bashtoken_bleu.append((vocab_size(mytransformer),bleu[4]))

plt.legend(loc='upper left', frameon=True)
plt.grid(True)

def run(vsize:int):
  def mysubtok(m):
    def _config(d):
      d['target_vocab_size']=vsize
      d['vocab_file'] = [promise, 'vocab.%d' % vsize]
      d['train_data_min_count']=None
      d['file_byte_limit'] = 1e6 if vsize > 5000 else 1e5
      return mkconfig(d)
    return redefine(all_nl2bashsubtok,_config)(m,
      shuffle=True, with_bash_charset=False, with_bash_subtokens=False)

  def mytransformer(m):
    def _config(c):
      c['train_steps']=6*5000
      c['params']['beam_size']=3 # As in Tellina paper
      return mkconfig(c)
    return redefine(transformer_wmt,_config)(m, mysubtok(m))

  return mysubtok, mytransformer

plt.figure(3)
plt.xlabel("Epoches")
plt.title("BLEU, Changing vocabulary size of Baseline model")

plt.plot(range(len(unshuffled_bleu)), unshuffled_bleu, label=f'Unshuffled transformer', color='red')
plt.plot(range(len(baseline_bleu)), baseline_bleu, label=f'Baseline transformer', color='orange')

out=Path(join(environ['STAGEDML_ROOT'],'_experiments','nl2bash','vsizebl'))
makedirs(out, exist_ok=True)
for i,vsize in enumerate([ 15000, 10000, 5000, 1700 ]) :
  mysubtok,mytransformer=run(vsize)
  rref=realize(instantiate(mytransformer))
  mksymlink(rref, out, str(vsize), withtime=False)
  bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
  plt.plot(range(len(bleu)), bleu, label=f'vsize-{vocab_size(mytransformer)}')
  summary_baseline_bleu.append((vocab_size(mytransformer),bleu[4]))

plt.legend(loc='upper left', frameon=True)
plt.grid(True)

def run2(vsize:int):
  def mysubtok(m):
    def _config(d):
      d['target_vocab_size']=vsize
      d['vocab_file'] = [promise, 'vocab.%d' % vsize]
      d['train_data_min_count']=None
      d['file_byte_limit'] = 1e6 if vsize > 5000 else 1e5
      return mkconfig(d)
    return redefine(all_nl2bashsubtok,_config)(m,
      shuffle=True, with_bash_charset=True, with_bash_subtokens=True)

  def mytransformer(m):
    def _config(c):
      c['train_steps']=6*5000
      c['params']['beam_size']=3 # As in Tellina paper
      return mkconfig(c)
    return redefine(transformer_wmt,_config)(m, mysubtok(m))

  return mysubtok, mytransformer

plt.figure(3)
plt.xlabel("Epoches")
plt.title("BLEU, Changing vocabulary size")

plt.plot(range(len(unshuffled_bleu)), unshuffled_bleu, label=f'Unshuffled transformer', color='red')
plt.plot(range(len(baseline_bleu)), baseline_bleu, label=f'Baseline transformer', color='orange')

out=Path(join(environ['STAGEDML_ROOT'],'_experiments','nl2bash','vsize'))
makedirs(out, exist_ok=True)
for i,vsize in enumerate([ 15000, 10000, 5000, 1700 ]) :
  mysubtok,mytransformer=run2(vsize)
  rref=realize(instantiate(mytransformer))
  mksymlink(rref, out, str(vsize), withtime=False)
  bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
  plt.plot(range(len(bleu)), bleu, label=f'vsize-{vocab_size(mytransformer)}')
  summary_bashtoken_bleu.append((vocab_size(mytransformer),bleu[4]))

plt.legend(loc='upper left', frameon=True)
plt.grid(True)

def singlechar_subtok(m):
  vsize=10000
  def _config(d):
    d['target_vocab_size']=vsize
    d['vocab_file'] = [promise, 'vocab.%d' % vsize]
    d['no_slave_multichar'] = True
    d['train_data_min_count']=None
    return mkconfig(d)
  return redefine(all_nl2bashsubtok,_config)(m)

def singlechar_transformer(m):
  def _config(c):
    c['train_steps']=6*5000
    c['params']['beam_size']=3 # As in Tellina paper
    return mkconfig(c)
  return redefine(transformer_wmt,
                  new_config=_config,
                  new_matcher=match_some())(m, singlechar_subtok(m), num_instances=5)

plt.figure(4)
plt.xlabel("Epoches")
plt.title("BLEU, Single-char punctuation tokens")

plt.plot(range(len(unshuffled_bleu)), unshuffled_bleu, label=f'Unshuffled transformer', color='red')
plt.plot(range(len(baseline_bleu)), baseline_bleu, label=f'Baseline transformer', color='orange')

out=Path(join(environ['STAGEDML_ROOT'],'_experiments','nl2bash','singlechar'))
makedirs(out, exist_ok=True)
summary_1punct_bleu=[]
for i,rref in enumerate(realizeMany(instantiate(singlechar_transformer))):
  mksymlink(rref, out, f'run-{i}', withtime=False)
  bleu=read_tensorflow_log(join(rref2path(rref),'eval'), 'bleu_cased')
  plt.plot(range(len(bleu)), bleu, label=f'run-{vocab_size(singlechar_transformer)}', color='blue')
  summary_1punct_bleu.append((vocab_size(singlechar_transformer),bleu[4]))

plt.legend(loc='upper left', frameon=True)
plt.grid(True)

plt.figure(5)
plt.xlabel("Vocab size")
plt.title("BLEU per vocab_size")
def _plotdots(ds, **kwargs):
  plt.plot(*zip(*ds), marker='o', ls='', **kwargs)
_plotdots(summary_baseline_bleu, label=f'Baseline')
_plotdots(summary_unshuffled_bleu, label=f'Unshuffled')
_plotdots(summary_bashtoken_bleu, label=f'Bashtoken')
_plotdots(summary_1punct_bleu, label=f'Bashtoken+1puct')
plt.legend(loc='upper right', frameon=True)
plt.grid(True)
