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
    parser.add_argument('labels', default=None, type=str)
    parser.add_argument('--diff', action='store_true')
    parser.add_argument('--alternative', default=None, type=str)
    parser.add_argument('--output-all', action='store_true')


    args = parser.parse_args()

    if args.labels is not None:
        with open(args.labels, mode='r', encoding='utf-8') as fh:
            labels = [int(x.strip()) for x in fh]
    else:
        labels = []

    import cPickle as pickle

    alternatives = []
    if args.alternative is not None:
        f = open(args.alternative, 'rb')
        alternatives = pickle.load(f)
        f.close()

    f = open(args.predfile, 'rb')
    predictions = pickle.load(f)
    f.close()

    def get_output(pred, alt = None):
        assert len(pred[0]) == 1 and len(pred[1]) == 1 and len(pred[2]) == 1
        # distance between embeddings
        if args.diff:
            import scipy.spatial
            dist_h1_ref = ((pred[0][0] - pred[2][0]) ** 2).sum()
            dist_h2_ref = ((pred[1][0] - pred[2][0]) ** 2).sum()
            dist_h1_h2_diff = dist_h1_ref - dist_h2_ref
            dist_h1_h2 = ((pred[1][0] - pred[0][0]) ** 2).sum()
            cos_h1_ref = scipy.spatial.distance.cosine(pred[0][0], pred[2][0])
            cos_h2_ref = scipy.spatial.distance.cosine(pred[1][0], pred[2][0])
            cos_h1_h2_diff = cos_h1_ref - cos_h2_ref
            cos_h1_h2 = scipy.spatial.distance.cosine(pred[1][0], pred[0][0])
            unnormalized_cos_h1_ref = (pred[0][0] * pred[2][0]).sum()
            unnormalized_cos_h2_ref = (pred[1][0] * pred[2][0]).sum()
            unnormalized_cos_h1_h2 = (pred[1][0] * pred[0][0]).sum()
            if alt is not None:
                to_append = [x[0] for x in alt]
            else:
                to_append = []
            to_append.extend([dist_h1_ref, dist_h2_ref, dist_h1_h2, cos_h1_ref, cos_h2_ref, cos_h1_h2,])
            to_append.extend([dist_h1_h2_diff, cos_h1_h2_diff])
            to_append.extend([unnormalized_cos_h1_ref, unnormalized_cos_h2_ref, unnormalized_cos_h1_h2])
            combined_array = 1000 * np.hstack([pred[0][0], pred[1][0], pred[2][0], pred[0][0] - pred[2][0], pred[1][0] - pred[2][0]] + to_append)
            # combined_array = 1000 * np.hstack(to_append)
        else:
            combined_array = 1000 * np.hstack([pred[0][0], pred[1][0], pred[2][0]])
        output = u' '.join([u'{}:{}'.format(idx + 1, x) for idx, x in enumerate(combined_array)])
        return output

    # assert len(labels)==len(predictions)
    if len(labels) < len(predictions):
        remain = [1] * (len(predictions) - len(labels))
        labels = labels + remain
    dur = len(predictions)
    min_pred = predictions[:dur]
    min_labels = labels[:dur]
    for idx, (pred, label) in enumerate(zip(min_pred, min_labels)):
        if len(alternatives) > 0:
            output = get_output(pred, alt = alternatives[idx])
        else:
            output = get_output(pred)
        print u'{} {}'.format(label, output)


if __name__ == '__main__':
    main()
