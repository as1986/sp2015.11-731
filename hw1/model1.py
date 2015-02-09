#!/usr/bin/python

from collections import Counter, defaultdict, namedtuple
from io import open
import random

Position = namedtuple("Position", "idx len_g len_e")

def init_align(numbered_e_sentences, numbered_g_sentences):
    # counts of alignments g -> *
    g_counts = defaultdict(int)

    # counts of alignments g -> e
    g_e_counts = defaultdict(lambda: defaultdict(int))

    # counts of positions g_idx, len_g, len_e -> e_idx
    position_counts = defaultdict(lambda: defaultdict(int))

    alignments = []
    for g_sen, e_sen in zip(numbered_g_sentences,numbered_e_sentences):
        alignment = []
        for e_idx, e in enumerate(e_sen):
            g_idx = random.randint(-1, len(g_sen)-1)
            alignment.append(g_idx)
            if g_idx == -1:
                g_counts[-1] += 1
                g_e_counts[-1][e] += 1
            else:
                g_counts[g_sen[g_idx]] += 1
                g_e_counts[g_sen[g_idx]][e] += 1
            position_counts[Position(idx=g_idx, len_g=len(g_sen), len_e=len(e_sen))][e_idx] += 1
            assert(position_counts[Position(idx=g_idx, len_g=len(g_sen), len_e=len(e_sen))][e_idx]>0)
        alignments.append(alignment)
    print position_counts[Position(idx=0, len_g=17, len_e=18)][0]
    return (alignments, g_counts, g_e_counts, position_counts)


def sample_from(table):
    summed = sum(table)
    assert(summed>0)
    draw = random.uniform(0., summed)
    current_sum = 0.
    for i in range(len(table)):
        current_sum += table[i]
        if current_sum > draw:
            return i
    # unlikely
    print "things have probably gone bad in the sampler..."
    return len(table)-1

def sample(alignment, g_sen, e_sen, g_counts, g_e_counts, positions, e_vocab_size, theta, beta=2.):

    def increment_count(g_idx, e_idx, g_len, e_len, e, by):
        if g_idx == -1:
            g_counts[-1] += by
            g_e_counts[-1][e] += by
        else:
            g_counts[g_sen[g_idx]] += by
            g_e_counts[g_sen[g_idx]][e] += by
        positions[Position(idx=g_idx, len_g=g_len, len_e = e_len)][e_idx] += by


    for e_idx, (g_idx,e) in enumerate(zip(alignment, e_sen)):
        # first remove count
        if (positions[Position(idx=g_idx, len_g=len(g_sen), len_e=len(e_sen))][e_idx]==0):
            print 'positions: idx: {} len_g: {} len_e: {} e_idx: {}'.format(g_idx, len(g_sen), len(e_sen), e_idx)
            assert(False)
        increment_count(g_idx, e_idx, len(g_sen), len(e_sen), e, -1)

    for a_idx, e in enumerate(e_sen):
        prob_table = []
        prob_g = (positions[Position(idx=-1, len_g=len(g_sen), len_e = len(e_sen))][a_idx] + beta) * (g_e_counts[-1][e] + theta) * 1. / (g_counts[-1] + theta * e_vocab_size )
        prob_table.append(prob_g)
        for g_idx, g in enumerate(g_sen):
            prob_g = (positions[Position(idx=g_idx, len_g=len(g_sen), len_e = len(e_sen))][a_idx] + beta) * (g_e_counts[g][e] + theta) * 1. / (g_counts[g] + theta * e_vocab_size )
            prob_table.append(prob_g)
        sampled = sample_from(prob_table) - 1
        alignment[a_idx] = sampled
    for e_idx, (g_idx,e) in enumerate(zip(alignment, e_sen)):
        increment_count(g_idx, e_idx, len(g_sen), len(e_sen), e, 1)
    return


def create_vocab(sentences):
    all_elems = dict()
    to_return = []
    for s in sentences:
        to_append = []
        for w in s:
            if w not in all_elems:
                all_elems[w] = len(all_elems)
            to_append.append(all_elems[w])
        to_return.append(to_append)
    return all_elems, to_return

def read_aligned(fname):
    f_sentences = []
    e_sentences = []
    with open(fname, encoding='utf-8', mode='r') as fh:
        for l in fh:
            loaded = l.strip().split(u'|||')
            # language f
            f_sentences.append(loaded[0].strip().split(u' '))
            # language f
            e_sentences.append(loaded[1].strip().split(u' '))
    assert len(f_sentences) == len(e_sentences)
    return (f_sentences, e_sentences)

def init_record(alignments):
    to_return = []
    for alignment in alignments:
        to_append = []
        for i in range(len(alignment)):
            to_append.append(Counter())
        to_return.append(to_append)
    return to_return

def record(alignments, recorded):
    # print 'recording'
    assert len(alignments) == len(recorded)
    for i in range(len(recorded)):
        assert len(alignments[i]) == len(recorded[i])
        for j in range(len(recorded[i])):
            recorded[i][j].update([alignments[i][j]])
    return

def output_record(recorded, epoch, great_epoch, theta, prefix):
    with open(u'{}_epoch_{}_great_epoch_{}_theta_{}'.format(prefix, epoch, great_epoch, theta), mode='w') as fh:
        for record in recorded:
            fh.write(u' '.join([unicode(x.most_common(1)[0][0]) for x in record])+u'\n')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('aligned')
    parser.add_argument('--output-prefix', default='output')
    args = parser.parse_args()
    (f,e) = read_aligned(args.aligned)
    f_vocab, numbered_f = create_vocab(f)
    e_vocab, numbered_e = create_vocab(e)
    # print len(f_vocab), len(e_vocab)

    rec = None

    # great epochs
    num_great_epochs = 3

    # epochs
    num_epochs = 100
    
    burnins = 10

    record_every = 10

    theta = 1e-5

    for great_epoch in range(num_great_epochs):
        (alignments, f_counts, f_e_counts, positions) = init_align(numbered_e,numbered_f)
        if rec is None:
            rec = init_record(alignments)
        for epoch in range(num_epochs):
            # print 'epoch {}:'.format(epoch)
            shuffled = range(len(alignments))
            random.shuffle(shuffled)
            for sen_idx in shuffled:
                sample(alignments[sen_idx], numbered_f[sen_idx], numbered_e[sen_idx], f_counts, f_e_counts, positions, len(e_vocab), theta)
            if epoch + 1 > burnins:
                record(alignments, rec)
                if epoch % record_every == 0:
                    output_record(rec, epoch, great_epoch, theta, args.output_prefix)
    return

if __name__ == '__main__':
    main()
