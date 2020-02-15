"""
This file contains a collection of aliases for Derivation references. They help
Mypy the static typechecker to catch errors early.
"""

from pylightnix import DRef

class BertCP(DRef):
  pass

class Glue(DRef):
  pass

class Squad11(DRef):
  pass

class GlueTFR(DRef):
  pass

class Squad11TFR(DRef):
  pass

class BertGlue(DRef):
  pass

class BertSquad(DRef):
  pass

class NL2Bash(DRef):
  pass
