from pylightnix import Manager, mknode, fetchurl, promise, mklens
from stagedml.types import Squad11

def fetchsquad11(m:Manager)->Squad11:
  trainref = fetchurl(m,
      name='squad11-train',
      url='https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
      sha256='3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955',
      mode='',
      output=[promise,'train-v1.1.json'])
  devref = fetchurl(m,
      name='squad11-dev',
      url='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
      sha256='95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9',
      mode='',
      output=[promise,'dev-v1.1.json'])
  return Squad11(mknode(m, {
    'name':'squad11',
    'train': mklens(trainref).output.refpath,
    'dev': mklens(devref).output.refpath
    }))

