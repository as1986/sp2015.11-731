'''
calculates meteor parameters
'''

__author__ = 'as1986'

import argparse
from io import open
from mtproject.toy import load_sentences
import numpy as np

def get_ngrams(sentence, n=1):
    to_return = set()
    for i in xrange(len(sentence) - n + 1):
        to_return.add(tuple(sentence[i:i+n]))
    return to_return

def get_precision(h, ref):
    return float(len(h&ref)+1e-5) / (len(h)+1e-5)

def get_recall(h, ref):
    return float(len(h&ref)+1e-5) / (len(ref)+1e-5)

def f(precision, recall, beta=1):
    sum = (beta * beta) * precision + recall
    if sum == 0:
        return 0
    return (1 + beta * beta) * precision * recall / sum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('--beta', default=1, type=float)
    args = parser.parse_args()

    # we create a generator and avoid loading all sentences into a list
    def sentences(fname, infinite=False):
        with open(fname, encoding='utf-8', mode='r') as f:
            for pair in f:
                yield pair.split(u' ||| ')
        if infinite:
            while True:
                yield ['', '', '']

    to_save = []
    vocab = dict()
    for (s1, s2, sref) in sentences(args.infile):
        vocab, (h1,h2,ref) = load_sentences([s1,s2,sref], using_vocab=vocab)
        # unigrams = [get_ngrams(h1, 1), get_ngrams(h2, 1), get_ngrams(ref, 1)]
        bigrams = [get_ngrams(h1, 2), get_ngrams(h2, 2), get_ngrams(ref, 2)]
        trigrams = [get_ngrams(h1, 3), get_ngrams(h2, 3), get_ngrams(ref, 3)]
        h1_precision = get_precision(bigrams[0] | trigrams[0], bigrams[2] | trigrams[2])
        h2_precision = get_precision(bigrams[1] | trigrams[1], bigrams[2] | trigrams[2])
        h1_recall = get_recall(bigrams[0] | trigrams[0], bigrams[2] | trigrams[2])
        h2_recall = get_recall(bigrams[1] | trigrams[1], bigrams[2] | trigrams[2])
        h1_f = f(h1_precision,h1_recall, beta=args.beta)
        h2_f = f(h2_precision,h2_recall, beta=args.beta)
        each = np.asarray([[[h1_precision, h1_recall, h1_f]], [[h2_precision, h2_recall, h2_f]], [[len(ref)]]])
        to_save.append(each)

    import cPickle as pickle
    output = file(args.outfile, 'wb')
    print 'saving to file {}'.format(args.outfile)
    pickle.dump(to_save, output, protocol=pickle.HIGHEST_PROTOCOL)
    output.close()

if __name__ == '__main__':
    main()
