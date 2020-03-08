from pylightnix import ( RefPath, Build, Path, Config, Manager, RRef, DRef,
    Context, build_wrapper, build_path, build_outpath, build_cattrs, mkdrv,
    rref2path, mkbuild, mkconfig, match_only, instantiate, realize, lsref,
    catref, store_cattrs, get_executable, fetchurl, mknode, checkpaths, mklens,
    promise, match_latest, forcelink, relpath )

from stagedml.utils import ( flines )
from stagedml.types import ( Optional, Any, List, Tuple, Union, Dict )
from stagedml.imports.sys import ( join, random )


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


def lineshuffle(m:Manager, src:Dict[str,RefPath])->DRef:
  config={'name':'shuffle','src':src}
  firstpath=None
  for nm,rp in src.items():
    firstpath=rp if firstpath is None else firstpath
    config.update({nm:[promise,f"{nm}.txt"]})
  assert firstpath is not None
  def _realize(b:Build)->None:
    print(f'Shuffling {src.values()}')
    build_outpath(b)
    nlines=flines(mklens(firstpath,b=b).syspath)
    ids=[random() for _ in range(nlines)]
    for nm,rp in src.items():
      with open(mklens(rp,b=b).syspath,'r') as f:
        lines=f.readlines()
      assert len(lines)==nlines
      with open(mklens(b).get(nm).syspath,'w') as f:
        f.writelines(map(lambda x:x[1], sorted(zip(ids,lines))))
  return mkdrv(m, mkconfig(config), match_latest(), build_wrapper(_realize))


# def linkfiles(m:Manager, src:Dict[str,RefPath], name:str='linkfiles')->DRef:
#   """ FIXME: figure out how to deal with symlinks from tmp->storage """
#   config={'name':name, 'src':src}
#   for nm in src.keys():
#     config.update({nm:[promise,nm]})
#   def _realize(b:Build)->None:
#     o=build_outpath(b)
#     for nm,rp in src.items():
#       forcelink(Path(relpath(mklens(rp,b=b).syspath, o)), mklens(b).get(nm).syspath)
#   return mkdrv(m, mkconfig(config), match_latest(), build_wrapper(_realize))


