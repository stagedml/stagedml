"""
This file contains a collection of aliases for Derivation references. They help
Mypy the static typechecker to catch errors early.
"""

from typing import ( Optional, Dict, Any, List, Tuple, Union, Callable, Set,
    Iterable, NamedTuple, NewType, TypeVar )

from pylightnix import DRef

BertCP=NewType('BertCP',DRef)

Glue=NewType('Glue',DRef)

#: Reference to the RuSentiment dataset
Rusent=NewType('Rusent',DRef)

class BertFinetuneTFR(DRef):
  pass

Squad11=NewType('Squad11',DRef)

class GlueTFR(BertFinetuneTFR):
  pass

class Squad11TFR(BertFinetuneTFR):
  pass

BertGlue=NewType('BertGlue',DRef)

BertSquad=NewType('BertSquad',DRef)

NL2Bash=NewType('NL2Bash',DRef)

WmtSubtok=NewType('WmtSubtok',DRef)

TransWmt=NewType('TransWmt',DRef)

Mnist=NewType('Mnist',DRef)

ConvnnMnist=NewType('ConvnnMnist',DRef)

Tr2Subtok=NewType('Tr2Subtok',DRef)

Trans2=NewType('Trans2',DRef)

#! Raw dump of wikipedia files """
Wikidump=NewType('Wikidump',DRef)

#! Wikipedia text, extracted with WikiExtractor """
Wikitext=NewType('Wikitext',DRef)

#! Wikipedia data in TensorFlow Records format """
WikiTFR=NewType('WikiTFR',DRef)

#! Reference to the pre-trained BERT model
BertPretrain=NewType('BertPretrain',BertCP)

