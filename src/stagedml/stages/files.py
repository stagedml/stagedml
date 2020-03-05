from pylightnix import ( RefPath, Build, Path, Config, Manager, RRef, DRef,
    Context, build_wrapper, build_path, build_outpath, build_cattrs, mkdrv,
    rref2path, mkbuild, mkconfig, match_only, instantiate, realize, lsref,
    catref, store_cattrs, get_executable, fetchurl, mknode, checkpaths, mklens,
    promise )

from stagedml.utils import ( flines )
from stagedml.types import ( Optional, Any, List, Tuple, Union, Dict )
from stagedml.imports import ( join )


def catfiles(m:Manager, files:List[RefPath], outname:Optional[str]=None)->DRef:
  """ Concatenate `files` into a single file named `outname` (defaults to
  'result') """
  outname_='result' if outname is None else outname
  def _realize(b:Build)->None:
    o=build_outpath(b)
    with open(mklens(b).output.syspath,'w') as dst:
      for rp in files:
        path=build_path(b,rp)
        nwritten=0
        with open(path,'r',newline='\n') as tgt:
          for line in tgt:
            dst.write(line.strip())
            dst.write('\n')
            nwritten+=1
        print(f'Written {nwritten} lines from {rp}')

  return mkdrv(m,mkconfig({'version':3, 'files':files, 'output':[promise,outname_]}),
                 match_only(),
                 build_wrapper(_realize))


def splitfile(m:Manager, src:RefPath, fractions:List[Tuple[str,str,float]])->DRef:
  attrs=[an for an,_,_ in fractions]
  names=[nm for _,nm,_ in fractions]
  ratios=[rt for _,_,rt in fractions]
  def _config()->Config:
    cfg={
      'name':'splitfile',
      'source':src,
      'ratios':ratios,
      'names':names,
      'version':2
    }
    for an,nm in zip(attrs,names):
      cfg.update({an:[promise,nm]})
    return mkconfig(cfg)

  def _realize(b:Build)->None:
    o=build_outpath(b)
    srcpath=mklens(b).source.syspath
    nlines=flines(srcpath)

    def _iter():
      with open(srcpath,'r',newline='\n') as f:
        for _,l in enumerate(f):
          yield l.strip()

    def _write(i, dstpath:str, nlines:int):
      with open(dstpath,'w') as f:
        for _ in range(nlines):
          l=next(i)
          f.write(l); f.write('\n')
    try:
      g=_iter()
      for an,nm,ratio in zip(attrs,names,ratios):
        _write(g, mklens(b).get(an).syspath, int(ratio*nlines))
    except StopIteration:
      pass

  return mkdrv(m, _config(), match_only(), build_wrapper(_realize))


