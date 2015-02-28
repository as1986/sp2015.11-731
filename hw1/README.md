I implemented a Bayesian variant of IBM Model 2 with Dirichlet priors. I used Gibbs sampling from 3 independent chains.
The first 10 samples are discarded and the rest 80 samples are collected. I ran Gibbs sampling on both de-en and en-de.
I also implemented the `grow-diag` algorithm as described [here](http://www.statmt.org/moses/?n=FactoredTraining.AlignWords).
`output.txt` contains the output from `grow_diag.py`.

I was trying to split German words but the samples collected thus far do not seem have converged. So I am reporting the results
from Model 2 (which is actually implemented in `model1.py`.)


There are three Python programs here (`-h` for usage):

 - `./align` aligns words using Dice's coefficient.
 - `./check` checks for out-of-bounds alignment points.
 - `./grade` computes alignment error rate.

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./align -t 0.9 -n 1000 | ./check | ./grade -n 5


The `data/` directory contains a fragment of the German/English Europarl corpus.

 - `data/dev-test-train.de-en` is the German/English parallel data to be aligned. The first 150 sentences are for development; the next 150 is a blind set you will be evaluated on; and the remainder of the file is unannotated parallel data.

 - `data/dev.align` contains 150 manual alignments corresponding to the first 150 sentences of the parallel corpus. When you run `./check` these are used to compute the alignment error rate. You may use these in any way you choose. The notation `i-j` means the word at position *i* (0-indexed) in the German sentence is aligned to the word at position *j* in the English sentence; the notation `i?j` means they are "probably" aligned.

