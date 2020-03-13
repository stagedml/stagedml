NL2Bash experiment
==================

This folder contains the resuslts of applying [TensorFlow official Transformer
model](https://github.com/tensorflow/models/tree/master/official/nlp/transformer)
to the NL2Bash dataset described in the [NL2Bash: A Corpus and Semantic Parser
for Natural Language Interface to the Linux Operating
System](https://arxiv.org/abs/1802.08979).


The report demonstrates how StagedML library could be used to simplify the
process of carrying out ML experiments. In particular, we want to highlight the
following facts:

1. The final version of the top-level module is only a one-screen long. At the
   same time, it allows the programmer to change every aspect of the experiment.
   The code connects the primitive stages together and tweak some parameters.
2. StagedML caches the results of running stages by creating immutable objects
   in the Pylightnix storage. The collection of such objects is accessible to
   users via Pylightnix API.
3. The final report may be generated from scrath by running a `Makefile` in this
   directory. The steps of downloading datasets and training the models is
   handled by the StagedML. (Installing StagedML systemwide does require running
   additional `sudo -H make install` command inside the Docker container)
