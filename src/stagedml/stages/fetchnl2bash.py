from pylightnix import ( Hash, RefPath, Build, Path, Config, Manager, RRef,
    DRef, Context, build_wrapper, build_path, build_outpath, build_cattrs,
    mkdrv, rref2path, mkbuild, mkconfig, match_only, instantiate, realize,
    lsref, catref, store_cattrs, get_executable, dirhash )

from stagedml.utils.files import ( system, flines )

from stagedml.imports import ( environ, join, basename, dedent, contextmanager,
    isfile )

from stagedml.types import ( Optional,Any,List,Tuple,Union, NL2Bash )



NL2BASH_ROOT=environ.get('NL2BASH_ROOT', join('/','workspace','3rdparty','nl2bash_essence', 'src'))

def fetchnl2bash(m:Manager)->DRef:
  def _config()->Config:
    sha256='a1bfa30d4b979cabf507df4a139a20e20634f6978669186aac57b3378263fa73'
    train_fraction=0.8
    dev_fraction=0.1
    test_fraction=0.1
    version=2
    return mkconfig(locals())
  def _realize(b:Build)->None:
    o=build_outpath(b)
    c=build_cattrs(b)
    nl_path=join(o,'all.nl')
    cm_path=join(o,'all.cm')
    system(['cp', f'{NL2BASH_ROOT}/data/bash/all.nl', nl_path])
    system(['cp', f'{NL2BASH_ROOT}/data/bash/all.cm', cm_path])
    h=dirhash(o)
    assert h==Hash(c.sha256), f"Sha256 mismatch. Expected '{c.sha256}', got '{h}'"
    nlines=flines(nl_path)
    assert nlines==flines(cm_path)

    def _iter():
      with open(nl_path,'r',newline='\n') as f1, \
           open(cm_path,'r',newline='\n') as f2:
        for _,(nl,cm) in enumerate(zip(f1,f2)):
          yield (nl.strip(),cm.strip())

    def _write(i, inputs, targets, n:int):
      with open(join(o,inputs),'w') as f1, \
           open(join(o,targets),'w') as f2:
        for _ in range(n):
          (nl,cm)=next(i)
          f1.write(nl); f1.write('\n')
          f2.write(cm); f2.write('\n')

    try:
      g=_iter()
      _write(g, join(o,'train.nl'),join(o,'train.cm'), int(c.train_fraction*nlines))
      _write(g, join(o,'dev.nl'),join(o,'dev.cm'), int(c.dev_fraction*nlines))
      _write(g, join(o,'test.nl'),join(o,'test.cm'), int(c.test_fraction*nlines))
    except StopIteration:
      pass

  return mkdrv(m, _config(), match_only(), build_wrapper(_realize))
