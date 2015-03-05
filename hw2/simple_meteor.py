'''
calculates meteor parameters
'''

__author__ = 'as1986'

import argparse
from io import open
from mtproject.toy import load_sentences

def get_ngrams(sentence, n=1):
    to_return = set()
    for i in xrange(len(sentence) - n + 1):
        to_return.add(tuple(sentence[i:i+n]))
    return to_return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    args = parser.parse_args()

    # we create a generator and avoid loading all sentences into a list
    def sentences(fname, infinite=False):
        with open(fname, encoding='utf-8', mode='r') as f:
            for pair in f:
                yield pair.split(u' ||| ')
        if infinite:
            while True:
                yield ['', '', '']

    for (h1, h2, ref) in sentences(args.infile):
        uni_h1 = get_ngrams(h1, 1)
        print uni_h1


if __name__ == '__main__':
    main()