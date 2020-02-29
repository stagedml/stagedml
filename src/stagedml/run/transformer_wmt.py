from stagedml.stages.all import *
from stagedml.stages.transformer_wmt import *

rref=realize(instantiate(all_transformer_wmtenru))
print(rref)

# b=repl_build()
# build(b)
