
from stagedml.stages.all import *
from stagedml.stages.transformer_wmt import *


repl_realize(instantiate(all_transformer_wmtenru))

b=repl_build()

# build(b)
