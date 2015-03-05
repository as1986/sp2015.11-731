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
    parser.add_argument('--diff', action='store_true')


    args = parser.parse_args()

    with open(args.labels, mode='r', encoding='utf-8') as fh:
        labels = [int(x.strip()) for x in fh]

    import cPickle as pickle

    f = open(args.predfile, 'rb')
    predictions = pickle.load(f)
    f.close()

    # assert len(labels)==len(predictions)
    dur = min([len(labels), len(predictions)])
    min_pred = predictions[:dur]
    min_labels = labels[:dur]
    for (pred, label) in zip(min_pred, min_labels):
        assert len(pred[0]) == 1 and len(pred[1]) == 1 and len(pred[2]) == 1
        # distance between embeddings
        if args.diff:
            import scipy.spatial
            dist_h1_ref = ((pred[0][0] - pred[2][0]) ** 2).sum()
            dist_h2_ref = ((pred[1][0] - pred[2][0]) ** 2).sum()
            dist_h1_h2 = ((pred[1][0] - pred[0][0]) ** 2).sum()
            cos_h1_ref = scipy.spatial.distance.cosine(pred[0][0], pred[2][0])
            cos_h2_ref = scipy.spatial.distance.cosine(pred[1][0], pred[2][0])
            cos_h1_h2 = scipy.spatial.distance.cosine(pred[1][0], pred[0][0])
            combined_array = np.hstack([dist_h1_ref, dist_h2_ref, dist_h1_h2, cos_h1_ref, cos_h2_ref, cos_h1_h2])
            # combined_array = np.hstack([pred[0][0], pred[1][0], pred[2][0], dist_h1_ref, dist_h2_ref, dist_h1_h2])
        else:
            combined_array = np.hstack([pred[0][0], pred[1][0], pred[2][0]])
        output = u' '.join([u'{}:{}'.format(idx + 1, x) for idx, x in enumerate(combined_array)])
        print u'{} {}'.format(label, output)


if __name__ == '__main__':
    main()
