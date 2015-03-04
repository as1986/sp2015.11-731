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

    assert len(labels)==len(predictions)
    for (pred, label) in zip(predictions, labels):
        assert len(pred[0]) == 1 and len(pred[1]) == 1 and len(pred[2]) == 1
        combined_array = np.hstack([pred[0][0], pred[1][0], pred[2][0]])
        output = u' '.join([u'{}: {}'.format(idx + 1, x) for idx, x in enumerate(combined_array)])
        print u'{} {}'.format(label, output)


if __name__ == '__main__':
    main()
