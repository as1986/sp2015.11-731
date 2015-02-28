#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from mtproject.toy import load_sentences
from io import open
 
# DRY
def word_matches(h, ref):
    return sum(1 for w in h if w in ref)
    # or sum(w in ref for w in f) # cast bool -> int
    # or sum(map(ref.__contains__, h)) # ugly!
 
def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
            help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input, encoding='utf-8', mode='r') as f:
            for pair in f:
                # yield [sentence.strip().split() for sentence in pair.split(u' ||| ')]
                yield pair.split(u' ||| ')
 
    vocab = dict()
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        vocab, loaded = load_sentences([h1, h2, ref], vocab)
        s1 = loaded[0]
        s2 = loaded[1]
        sref = loaded[2]
        rset = set(sref)
        h1_match = word_matches(s1, rset)
        h2_match = word_matches(s2, rset)
        # print(-1 if h1_match > h2_match else # \begin{cases}
        #        (0 if h1_match == h2_match
        #            else 1)) # \end{cases}
 
    print len(vocab)
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
