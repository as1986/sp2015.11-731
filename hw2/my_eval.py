import argparse
from itertools import islice  # slicing for iterators
from io import open
from random import choice, shuffle

from mtproject.toy import load_sentences
import mtproject.deeplearning.layer
from mtproject.deeplearning.learning_rule import *
import mtproject.deeplearning.utils


def main():
    parser = argparse.ArgumentParser(description='Train LSTM.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
                        help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=2000, type=int,
                        help='Number of hypothesis pairs to evaluate')
    parser.add_argument('--labels', default='data/own-split.training.labels')
    parser.add_argument('--test-file', default=None, type=str)
    parser.add_argument('--save-every', default=1, type=int)
    parser.add_argument('--load-model', default=None, type=str)
    parser.add_argument('--predict', default=None, type=str)
    parser.add_argument('--embeddings', default='data/w2v_model', type=str)
    # note that if x == [2, 3, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()

    def load_embeddings(fname):
        import gensim

        to_return = gensim.models.Word2Vec.load(fname)
        return to_return

    def prepare_shared_embeddings(vocab, model):
        assert isinstance(vocab, dict)
        to_return = np.zeros((len(vocab) + 1, 100), dtype=np.float32)
        for k, v in vocab.iteritems():
            to_return[v] = model[k.lower()]
        return to_return

    # we create a generator and avoid loading all sentences into a list
    def sentences(fname, infinite=False):
        with open(fname, encoding='utf-8', mode='r') as f:
            for pair in f:
                # yield [sentence.strip().split() for sentence in pair.split(u' ||| ')]
                yield pair.split(u' ||| ')

    def labels(infinite=False):
        with open(opts.labels, encoding='utf-8', mode='r') as label_fh:
            for label in label_fh:
                yield int(label.strip())
        if infinite:
            while True:
                yield None

    vocab = dict()
    references = dict()
    pairs = dict()
    test_sentences = []
    if opts.test_file is not None:
        print 'loading test file {}'.format(opts.test_file)
        for (h1, h2, ref) in sentences(opts.test_file):
            vocab, loaded = load_sentences([h1, h2, ref], vocab)
            test_sentences.append((loaded[0], loaded[1], loaded[2]))

    references = dict()
    pairs = dict()
    idx = 0
    for (h1, h2, ref), label in islice(zip(sentences(opts.input, infinite=True), labels(infinite=True)),
                                       opts.num_sentences):
        if len(ref) == 0:
            # no more sentences
            print 'h1: {}, h2: {} label: '.format(h1, h2, label)
            raise Exception('more sentences than labels!')
        print 'idx: {}'.format(idx)
        idx += 1
        vocab, loaded = load_sentences([h1, h2, ref], vocab)
        s1 = loaded[0]
        s2 = loaded[1]
        sref = loaded[2]
        references[ref] = np.asarray([sref], dtype=np.int32)
        if ref not in pairs:
            pairs[ref] = []
        if label is None:
            # TODO basically ignore the test data now
            continue
        elif label == -1:
            pairs[ref].append((np.asarray([s1], dtype=np.int32), np.asarray([s2], dtype=np.int32)))
        elif label == 1:
            pairs[ref].append((np.asarray([s2], dtype=np.int32), np.asarray([s1], dtype=np.int32)))

    def save_model(model, fname):
        import cPickle as pickle

        f = file(fname, 'wb')
        print 'saving to file {}'.format(fname)
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_model(fname):
        import cPickle as pickle

        f = file(fname, 'rb')
        to_return = pickle.load(f)
        f.close()
        return to_return

    def prepare_nn(fname=None, predict_fname=None, shared_embeddings=None):
        print 'preparing nn...'
        if shared_embeddings is not None:
            print 'embeddings: {} {}'.format(len(shared_embeddings), len(shared_embeddings[0]))

        n_words = len(vocab)
        e_dim = 100
        lstm_dim = 50
        final_dim = 100
        x = T.imatrix('x')
        x_good = T.imatrix('x_good')
        x_bad = T.imatrix('x_bad')
        x_sane = T.imatrix('x_sane')

        layers = [
            mtproject.deeplearning.layer.tProjection(n_words,
                                                     e_dim) if shared_embeddings is None else mtproject.deeplearning.layer.tProjection(
                n_words, e_dim, embedding=shared_embeddings),
            mtproject.deeplearning.layer.LSTM(e_dim, lstm_dim, minibatch=True),
            mtproject.deeplearning.layer.LSTM(lstm_dim, lstm_dim, minibatch=True),
            mtproject.deeplearning.layer.LSTM(lstm_dim, final_dim, minibatch=True),
        ]
        if fname is not None:
            print 'loading model...'
            old_layers = load_model(fname)
            layers[0].copy_constructor(orig=old_layers[0])
            # TODO hacky
            layers[0].set_embeddings(shared_embeddings)
            layers[1].copy_constructor(orig=old_layers[1])
            layers[2].copy_constructor(orig=old_layers[2])
            layers[3].copy_constructor(orig=old_layers[3])


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

        def cos_dist(a, b):
            return 1 - (a * b).sum() / (a.norm(2) * b.norm(2))
        cost_good = cos_dist(y_good, y)
        cost_bad = cos_dist(y_bad, y)
        # cost_good_to_bad = cos_dist(y_bad, y_good)
        cost_sane = cos_dist(y_sane, y)

        cost_other_trans = theano.tensor.max([0, 1 - cost_sane])

        cost_hypotheses = theano.tensor.max([0, 1 + cost_good - cost_bad]) # + theano.tensor.max(
            # [0, 1 - cost_good_to_bad])
        cost = cost_hypotheses + cost_other_trans

        # L2
        for p in params:
            cost += 1e-4 * (p ** 2).sum()

        updates = learning_rule(cost, params, eps=1e-6, rho=0.65, method='adadelta')

        train = theano.function([x, x_good, x_bad, x_sane], [cost, y], updates=updates)
        sane_updates = learning_rule(cost_other_trans, params, eps=1e-6, rho=0.65, method='adadelta')
        unsupervised_train = theano.function([x, x_sane], [cost_other_trans, y], updates=sane_updates,
                                             on_unused_input='warn')
        predictor = theano.function([x], y)
        for r in xrange(2000):
            print 'round: {}'.format(r)
            seq = references.keys()
            shuffle(seq)
            for idx, ref in enumerate(seq):
                print 'idx: {}'.format(idx)
                random_ref = choice(references.keys())
                while random_ref == ref:
                    random_ref = choice(references.keys())
                random_sample = references[random_ref]
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
            # save_model(layers, 'layers_round_{}'.format(round))
            if predict_fname is not None and r % 10 == 0:
                print 'predicting...'
                to_output = []
                for idx, (good, bad, ref) in enumerate(test_sentences):
                    print 'predicting #{}'.format(idx)
                    emb_good = predictor([good])
                    emb_bad = predictor([bad])
                    emb_ref = predictor([ref])
                    to_output.append((emb_good, emb_bad, emb_ref))
                save_model(to_output, predict_fname+'_round_'.format(r))


    embeddings = load_embeddings(opts.embeddings)
    shared = prepare_shared_embeddings(vocab, model=embeddings)
    prepare_nn(fname=opts.load_model, predict_fname=opts.predict, shared_embeddings=shared)


# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
