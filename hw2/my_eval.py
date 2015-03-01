#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from mtproject.toy import load_sentences
from io import open
import theano.tensor as T
import theano
import numpy as np
import mtproject.deeplearning.layer
from mtproject.deeplearning.learning_rule import *
import mtproject.deeplearning.utils
 
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
    parser.add_argument('--labels', default='data/own-split.training.labels')
    parser.add_argument('--save-every', default=50, type=int)
    # note that if x == [2, 3, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input, encoding='utf-8', mode='r') as f:
            for pair in f:
                # yield [sentence.strip().split() for sentence in pair.split(u' ||| ')]
                yield pair.split(u' ||| ')

    def labels():
        with open(opts.labels, encoding='utf-8', mode='r') as label_fh:
            for label in label_fh:
                yield int(label.strip())
 
    vocab = dict()
    # note: the -n option does not work in the original code
    from collections import defaultdict
    references = dict()
    good = defaultdict(list)
    bad = defaultdict(list)
    for (h1, h2, ref), label in islice(zip(sentences(), labels()), opts.num_sentences):
        vocab, loaded = load_sentences([h1, h2, ref], vocab)
        s1 = loaded[0]
        s2 = loaded[1]
        sref = loaded[2]
        # h1_match = word_matches(s1, rset)
        # h2_match = word_matches(s2, rset)
        # print(-1 if h1_match > h2_match else # \begin{cases}
        #        (0 if h1_match == h2_match
        #            else 1)) # \end{cases}
        references[ref] = np.asarray([sref,], dtype=np.int32)
        if label == -1:
            good[ref].append(np.asarray([s1], dtype=np.int32))
            bad[ref].append(theano.shared(s2))
        elif label == 1:
            bad[ref].append(theano.shared(s1))
            good[ref].append(np.asarray([s2], dtype=np.int32))
    
    def save_model(model, fname):
        import cPickle as pickle
        f = file(fname, 'wb')
        pickle.dump(model, f,  protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_model(fname):
        import cPickle as pickle
        f = file(fname, 'rb')
        to_return = pickle.load(f)
        f.close()
        return to_return

    def prepare_nn():
        n_words = len(vocab)
        e_dim = 25
        lstm_dim = 100
        x = T.imatrix('x')
        layers = [
                mtproject.deeplearning.layer.tProjection(n_words, e_dim),
                mtproject.deeplearning.layer.LSTM(e_dim, lstm_dim, minibatch=True),
        ]

        params = []
        for layer in layers:
            params += layer.params

        for idx, layer in enumerate(layers):
            if idx == 0:
                layer_out = layer.fprop(x)
            else:
                layer_out = layer.fprop(layer_out)
        y = layers[-1].h[-1]

        y_func = theano.function([x], y)
        fixed_holder = T.matrix(dtype=theano.config.floatX)
        cost = -((fixed_holder - y) ** 2).sum()
        neg_cost = -cost
        for round in xrange(10):
            print 'round: {}'.format(round)
            for idx, ref in enumerate(references.iterkeys()):
                if len(good[ref])==0:
                    continue
                sref = references[ref]
                # print sref.type
                # print sref
                target = y_func(sref)
                updates = learning_rule(cost, params, eps=1e-6, rho=0.65, method='adadelta')
                neg_updates = learning_rule(neg_cost, params, eps=1e-6, rho=0.65, method='adadelta')
                print 'idx: {}'.format(idx)
                train = theano.function([x], cost, updates=updates, givens=[(fixed_holder,target)])
                for good_example in good[ref]:
                    # print good_example.type
                    train(good_example)
                neg_train = theano.function([x], neg_cost, updates=neg_updates, givens=[(fixed_holder,target)])
                for bad_example in bad[ref]:
                    neg_train(bad_example)


                



    prepare_nn()
    

 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
