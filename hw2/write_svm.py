'''
takes prediction pickle and writes SVM output
'''

__author__ = 'as1986'
from io import open
import numpy as np


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('predfile')
    parser.add_argument('labels')

    args = parser.parse_args()

    with open(args.labels, mode='r', encoding='utf-8') as fh:
        labels = [int(x.strip()) for x in fh]

    import cPickle as pickle

    f = open(args.predfile, 'rb')
    predictions = pickle.load(f)
    f.close()

    for (h1, h2, ref) in zip(predictions, labels):
        assert len(h1) == 1 and len(h2) == 1 and len(ref) == 1
        combined_array = h1[0] + h2[0] + ref[0]
        output = u' '.join([u'{}: {}'.format(idx + 1, x) for idx, x in enumerate(combined_array)])
        print u'{} {}'.format(labels, output)


if __name__ == '__main__':
    main()