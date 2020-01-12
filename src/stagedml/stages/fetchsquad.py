from pylightnix import Ref, mknode
from stagedml.stages.fetchurl import fetchurl
from stagedml.utils.refs import Squad11
from stagedml.utils.instantiate import Options

def fetchsquad11(o:Options)->Squad11:
  trainref = fetchurl(
      name='squad11-train',
      url='https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
      sha256='3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955',
      mode='')
  devref = fetchurl(
      name='squad11-dev',
      url='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
      sha256='95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9',
      mode='')
  return Squad11(mknode({
    'name':'squad11',
    'train_refpath': [trainref,'train-v1.1.json'],
    'dev_refpath': [devref,'dev-v1.1.json'],
    }))