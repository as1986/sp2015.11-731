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
            a = random.randint(0, len(g_sen))
            alignment.append(a)
            if a == 0:
                g_counts[0] += 1
                g_e_counts[0][e] += 1
            else:
                g_idx = a - 1
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
    for a_idx, (a,e) in enumerate(zip(alignment, e_sen)):
        # first remove count
        if a == 0:
            g_counts[0] -= 1
            g_e_counts[0][e] -= 1
        else:
            g_idx = a - 1
            g_counts[g_sen[g_idx]] -= 1
            g_e_counts[g_sen[g_idx]][e] -= 1
        prob_table = []
        prob_table.append((g_e_counts[0][e] + theta) * 1. / (g_counts[0] + theta * (e_vocab_size-1) ))
        for g in g_sen:
            idx = g
            prob_g = (g_e_counts[idx][e] + theta) * 1. / (g_counts[idx] + theta * (e_vocab_size-1) )
            prob_table.append(prob_g)
        sampled = sample_from(prob_table)
        alignment[a_idx] = sampled
        if sampled == 0:
            g_counts[0] += 1
            g_e_counts[0][e] += 1
        else:
            g_idx = sampled - 1
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
                all_elems[w] = len(all_elems)+1
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


def record(alignments):
    print 'dummy -- should be recording'
    return


def main():
    import sys
    (f,e) = read_aligned(sys.argv[1])
    f_vocab, numbered_f = create_vocab(f)
    e_vocab, numbered_e = create_vocab(e)
    print len(f_vocab), len(e_vocab)
    (alignments, f_counts, f_e_counts) = init_align(numbered_e,numbered_f)

    # epochs
    num_epochs = 5
    burnins = 1
    for epoch in range(num_epochs):
        print 'epoch {}:'.format(epoch)
        shuffled = range(len(alignments))
        random.shuffle(shuffled)
        for sen_idx in shuffled:
            sample(alignments[sen_idx], numbered_f[sen_idx], numbered_e[sen_idx], f_counts, f_e_counts, len(e_vocab))
        if epoch + 1 > burnins:
            record(alignments)
    return

if __name__ == '__main__':
    main()
