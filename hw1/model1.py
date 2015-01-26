#!/usr/bin/python

from collections import Counter, defaultdict
from io import open
import random


def init_align(numbered_e_sentences, numbered_g_sentences):
    # counts of alignments g -> *
    g_counts = defaultdict(int)

    # counts of alignments g -> e
    g_e_counts = defaultdict(lambda: defaultdict(int))

    alignments = []
    for g_sen, e_sen in zip(numbered_g_sentences,numbered_e_sentences):
        alignment = []
        for e in e_sen:
            a = random.randint(0, len(g_sen)-1)
            alignment.append(a)
            g_idx = a
            g_counts[g_sen[g_idx]] += 1
            g_e_counts[g_sen[g_idx]][e] += 1
        alignments.append(alignment)
    return (alignments, g_counts, g_e_counts)


def sample_from(table):
    summed = sum(table)
    draw = random.uniform(0., summed)
    current_sum = 0.
    for i in range(len(table)):
        current_sum += table[i]
        if current_sum > draw:
            return i
    # unlikely
    return len(table)-1

def sample(alignment, g_sen, e_sen, g_counts, g_e_counts, e_vocab_size):

    theta = 1
    for g_idx,e in zip(alignment, e_sen):
        # first remove count
        g_counts[g_sen[g_idx]] -= 1
        g_e_counts[g_sen[g_idx]][e] -= 1

    for a_idx, e in enumerate(e_sen):
        prob_table = []
        for g_idx in g_sen:
            prob_g = (g_e_counts[g_idx][e] + theta) * 1. / (g_counts[g_idx] + theta * e_vocab_size )
            prob_table.append(prob_g)
        sampled = sample_from(prob_table)
        alignment[a_idx] = sampled
    for g_idx,e in zip(alignment, e_sen):
        g_counts[g_sen[g_idx]] += 1
        g_e_counts[g_sen[g_idx]][e] += 1
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

def output_record(recorded, epoch, great_epoch):
    with open(u'output_epoch_{}_great_epoch_{}'.format(epoch, great_epoch), mode='w') as fh:
        for record in recorded:
            fh.write(u' '.join([unicode(x.most_common(1)[0][0]) for x in record])+u'\n')

def main():
    import sys
    (f,e) = read_aligned(sys.argv[1])
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

    for great_epoch in range(num_great_epochs):
        (alignments, f_counts, f_e_counts) = init_align(numbered_e,numbered_f)
        if rec is None:
            rec = init_record(alignments)
        for epoch in range(num_epochs):
            # print 'epoch {}:'.format(epoch)
            shuffled = range(len(alignments))
            random.shuffle(shuffled)
            for sen_idx in shuffled:
                sample(alignments[sen_idx], numbered_f[sen_idx], numbered_e[sen_idx], f_counts, f_e_counts, len(e_vocab))
            if epoch + 1 > burnins:
                record(alignments, rec)
                if epoch % record_every == 0:
                    output_record(rec, epoch, great_epoch)
    return

if __name__ == '__main__':
    main()
