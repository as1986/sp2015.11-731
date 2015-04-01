Things I tried include

- monotonic decoder from two directions
- rewriting sequence of translated phrases
- beam search with estimated future cost
- greedy sentence-wise translation combination
- append `<s>` and `</s>` to both ends of output after certain amount of words have been decoded

sequence-rewriting + beam search with future cost + `<s>` appending is the best single model (achieved `-4817.353308` in score) however the best results are obtained by combining all models.

===

original description below

===

There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model

