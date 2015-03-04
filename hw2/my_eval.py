#!/usr/bin/env python
import argparse  # optparse is deprecated
from itertools import islice  # slicing for iterators
from io import open
from random import choice

from mtproject.toy import load_sentences
import mtproject.deeplearning.layer
from mtproject.deeplearning.learning_rule import *
import mtproject.deeplearning.utils


def main():
    parser = argparse.ArgumentParser(description='Train LSTM.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
                        help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
                        help='Number of hypothesis pairs to evaluate')
    parser.add_argument('--labels', default='data/own-split.training.labels')
    parser.add_argument('--test-file', default=None, type=str)
    parser.add_argument('--save-every', default=1, type=int)
    parser.add_argument('--load-model', default=None, type=str)
    parser.add_argument('--predict', default=None, type=str)
    # note that if x == [2, 3, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()

    # we create a generator and avoid loading all sentences into a list
    def sentences(fname):
        with open(fname, encoding='utf-8', mode='r') as f:
            for pair in f:
                # yield [sentence.strip().split() for sentence in pair.split(u' ||| ')]
                yield pair.split(u' ||| ')

    def labels():
        with open(opts.labels, encoding='utf-8', mode='r') as label_fh:
            for label in label_fh:
                yield int(label.strip())

    vocab = dict()
    test_sentences = []
    if opts.test_file is not None:
        for (h1, h2, ref) in sentences(opts.test_file):
            vocab, loaded = load_sentences([h1, h2, ref], vocab)
            test_sentences.append((loaded[0], loaded[1], loaded[2]))

    references = dict()
    pairs = dict()
    for (h1, h2, ref), label in islice(zip(sentences(opts.input), labels()), opts.num_sentences):
        vocab, loaded = load_sentences([h1, h2, ref], vocab)
        s1 = loaded[0]
        s2 = loaded[1]
        sref = loaded[2]
        # h1_match = word_matches(s1, rset)
        # h2_match = word_matches(s2, rset)
        # print(-1 if h1_match > h2_match else # \begin{cases}
        # (0 if h1_match == h2_match
        # else 1)) # \end{cases}
        references[ref] = np.asarray([sref, ], dtype=np.int32)
        if ref not in pairs:
            pairs[ref] = []
        if label == -1:
            pairs[ref].append((np.asarray([s1], dtype=np.int32), np.asarray([s2], dtype=np.int32)))
        elif label == 1:
            pairs[ref].append((np.asarray([s2], dtype=np.int32), np.asarray([s1], dtype=np.int32)))

    def save_model(model, fname):
        import cPickle as pickle

        f = file(fname, 'wb')
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_model(fname):
        import cPickle as pickle

        f = file(fname, 'rb')
        to_return = pickle.load(f)
        f.close()
        return to_return

    def prepare_nn(fname=None, predict_fname=None):

        n_words = len(vocab)
        e_dim = 100
        lstm_dim = 100
        x = T.imatrix('x')
        x_good = T.imatrix('x_good')
        x_bad = T.imatrix('x_bad')
        x_sane = T.imatrix('x_sane')

        layers = [
            mtproject.deeplearning.layer.tProjection(n_words, e_dim),
            mtproject.deeplearning.layer.LSTM(e_dim, lstm_dim, minibatch=True),
            mtproject.deeplearning.layer.LSTM(e_dim, lstm_dim, minibatch=True),
            mtproject.deeplearning.layer.LSTM(e_dim, lstm_dim, minibatch=True),
        ]
        if fname is not None:
            print 'loading model...'
            old_layers = load_model(fname)
            layers[0].copy_constructor(orig=old_layers[0])
            layers[1].copy_constructor(orig=old_layers[1])
            layers[2].copy_constructor(orig=old_layers[2])
            layers[3].copy_constructor(orig=old_layers[3])

        if predict_fname is not None:
            print 'predicting...'
            for idx, layer in enumerate(layers):
                if idx == 0:
                    layer_out = layer.fprop(x)
                else:
                    layer_out = layer.fprop(layer_out)

            y = layers[-1].h[-1]
            predictor = theano.function([x], y)
            to_output = []
            for (good, bad, ref) in test_sentences:
                (emb_good, emb_bad, emb_ref) = (predictor([good]), predictor([bad]), predictor([ref]))
                to_output.append((emb_good, emb_bad, emb_ref))
            save_model(to_output, predict_fname)
            return

        layers_good = [
            mtproject.deeplearning.layer.tProjection(orig=layers[0]),
            mtproject.deeplearning.layer.LSTM(orig=layers[1]),
            mtproject.deeplearning.layer.LSTM(orig=layers[2]),
            mtproject.deeplearning.layer.LSTM(orig=layers[3]),
        ]

        layers_bad = [
            mtproject.deeplearning.layer.tProjection(orig=layers[0]),
            mtproject.deeplearning.layer.LSTM(orig=layers[1]),
            mtproject.deeplearning.layer.LSTM(orig=layers[2]),
            mtproject.deeplearning.layer.LSTM(orig=layers[3]),
        ]

        layers_sane = [
            mtproject.deeplearning.layer.tProjection(orig=layers[0]),
            mtproject.deeplearning.layer.LSTM(orig=layers[1]),
            mtproject.deeplearning.layer.LSTM(orig=layers[2]),
            mtproject.deeplearning.layer.LSTM(orig=layers[3]),
        ]

        params = []

        for layer in layers:
            params += layer.params

        for idx, (layer, layer_good, layer_bad, layer_sane) in enumerate(
                zip(layers, layers_good, layers_bad, layers_sane)):
            if idx == 0:
                layer_out = layer.fprop(x)
                layer_out_good = layer_good.fprop(x_good)
                layer_out_bad = layer_bad.fprop(x_bad)
                layer_out_sane = layer_sane.fprop(x_sane)
            else:
                layer_out = layer.fprop(layer_out)
                layer_out_good = layer_good.fprop(layer_out_good)
                layer_out_bad = layer_bad.fprop(layer_out_bad)
                layer_out_sane = layer_sane.fprop(layer_out_sane)
        y = layers[-1].h[-1]
        y_good = layers_good[-1].h[-1]
        y_bad = layers_bad[-1].h[-1]
        y_sane = layers_sane[-1].h[-1]

        cost_good = ((y_good - y) ** 2).sum()
        cost_bad = ((y_bad - y) ** 2).sum()
        cost_sane = ((y_sane - y) ** 2).sum()

        sane_cost = theano.tensor.max([0, 2 - cost_sane])

        cost = theano.tensor.max([0, 1 + cost_good - cost_bad]) + sane_cost

        # L2
        for p in params:
            cost += 1e-4 * (p ** 2).sum()

        updates = learning_rule(cost, params, eps=1e-6, rho=0.65, method='adadelta')
        train = theano.function([x, x_good, x_bad, x_sane], [cost, y], updates=updates)
        unsupervised_train = theano.function([x, x_sane], [cost, y], updates=updates,
                                             givens=[(cost, 0.)])
        for round in xrange(10):
            print 'round: {}'.format(round)
            for idx, ref in enumerate(references.iterkeys()):
                print 'idx: {}'.format(idx)
                random_sample = np.random.permutation(references[ref])
                if len(pairs[ref]) == 0:
                    this_cost, this_y = unsupervised_train(references[ref], random_sample)
                else:
                    chosen = choice(pairs[ref])
                    good_example = chosen[0]
                    bad_example = chosen[1]
                    this_cost, this_y = train(references[ref], good_example, bad_example, random_sample)
                if idx % 50 == 0:
                    print 'this cost: {}'.format(this_cost)
                    print 'this y: '
                    print this_y
            save_model(layers, 'layers_round_{}'.format(round))


    prepare_nn(fname=opts.load_model, predict_fname=opts.predict)


# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
