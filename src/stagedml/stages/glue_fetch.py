from pylightnix import ( Config, build_outpath,
    Manager, match_only, mkdrv, build_wrapper, Build )

from stagedml.types import Glue
from stagedml.datasets.glue.download_glue_data import main as glue_main


def config()->Config:
  name='glue-data'
  version='2'
  return Config(locals())


def download(b:Build)->None:
  glue_main(['--data_dir', build_outpath(b), '--tasks','all'])


def fetch_glue(m:Manager)->Glue:
  return Glue(mkdrv(m, config(), match_only(), build_wrapper(download)))

