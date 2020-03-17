NL2Bash Transformer
===================

This document describes the results of training [TensorFlow official
Trnasformer][1] on the [NL2Bash][5] dataset introduced by Xi Victoria Lin
et al. in their paper [Nl2Bash: Corpus and Semantic Parser for Natural Language
Interface to the Linux Operating System][2].

The primary goal of this work is to demonstrate the features of [StagedML][3]
library which we used to run experiments and generate this report. The secondary
goal is to evaluate the NL2BASH dataset on the stock transformer model.

We would like to highlight the following facts:

1. Top-level code snippets are one-screen long. At the same time, they allow
   programmer to change any parameter of the experiment.
2. StagedML caches the results of running stages by creating immutable objects
   in the Pylightnix storage. Configurations and training results are
   accessible to users via [Pylightnix][4] API.
3. Consequently, the rendering of the experiment reports may be largely authomatized.
   In this work, we generated reports from scrath by running a `Makefile` in the
   [/run/nl2bash](/run/nl2bash) directory.


Report
======

[Report](./out/Report.md) ( [Source](./Report.md.in) )
