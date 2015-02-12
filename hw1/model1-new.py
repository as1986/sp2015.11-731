#!/usr/bin/python

from collections import Counter, defaultdict, namedtuple
from io import open
import random

Position = namedtuple("Position", "idx len_g len_e")

def init_align(numbered_e_sentences, numbered_g_sentences, back_e, back_g):
    # counts of alignments g -> *
    g_counts = defaultdict(int)

    # counts of alignments g -> e
    g_e_counts = defaultdict(lambda: defaultdict(int))

    # counts of positions g_idx, len_g, len_e -> e_idx
    position_counts = defaultdict(lambda: defaultdict(int))

    alignments = []
    for sen_idx, (g_sen, e_sen) in enumerate(zip(numbered_g_sentences,numbered_e_sentences)):
        alignment = []
        for e_idx, e in enumerate(e_sen):
            g_idx = random.randint(-1, len(g_sen)-1)
            alignment.append(g_idx)
            if g_idx == -1:
                g_counts[-1] += 1
                g_e_counts[-1][e] += 1
                position_counts[Position(idx=-1, len_g=len(back_g[sen_idx]), len_e=len(back_e[sen_idx]))][back_e[sen_idx][e_idx]] += 1
                assert(position_counts[Position(idx=-1, len_g=len(back_g[sen_idx]), len_e=len(back_e[sen_idx]))][back_e[sen_idx][e_idx]]>0)
            else:
                g_counts[g_sen[g_idx]] += 1
                g_e_counts[g_sen[g_idx]][e] += 1
                position_counts[Position(idx=back_g[sen_idx][g_idx], len_g=len(back_g[sen_idx]), len_e=len(back_e[sen_idx]))][back_e[sen_idx][e_idx]] += 1
                assert(position_counts[Position(idx=back_g[sen_idx][g_idx], len_g=len(back_g[sen_idx]), len_e=len(back_e[sen_idx]))][back_e[sen_idx][e_idx]]>0)
        alignments.append(alignment)
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

def sample(alignment, g_sen, e_sen, g_counts, g_e_counts, positions, e_vocab_size, back_g_row, back_e_row, theta, beta=2.):

    def increment_count(g_idx, e_idx, e, by):
        if g_idx == -1:
            g_counts[-1] += by
            g_e_counts[-1][e] += by
            positions[Position(idx=-1, len_g=len(back_g_row), len_e = len(back_e_row))][back_e_row[e_idx]] += by
        else:
            g_counts[g_sen[g_idx]] += by
            g_e_counts[g_sen[g_idx]][e] += by
            # positions[Position(idx=g_idx, len_g=g_len, len_e = e_len)][e_idx] += by
            positions[Position(idx=back_g_row[g_idx], len_g=len(back_g_row), len_e = len(back_e_row))][back_e_row[e_idx]] += by


    for e_idx, (g_idx,e) in enumerate(zip(alignment, e_sen)):
        # first remove count
        #if (positions[Position(idx=back_g[sen_idx][g_idx], len_g=len(back_g[sen_idx]), len_e=len(back_e[sen_idx]))][back_e[sen_idx][e_idx]]==0):
        #    print 'positions: idx: {} len_g: {} len_e: {} e_idx: {}'.format(g_idx, len(g_sen), len(e_sen), e_idx)
        #    assert(False)
        increment_count(g_idx, e_idx, e, -1)

    for a_idx, e in enumerate(e_sen):
        prob_table = []
        prob_g = (positions[Position(idx=-1, len_g=len(back_g_row), len_e = len(back_e_row))][back_e_row[a_idx]] + beta) * (g_e_counts[-1][e] + theta) * 1. / (g_counts[-1] + theta * e_vocab_size )
        prob_table.append(prob_g)
        for g_idx, g in enumerate(g_sen):
            prob_g = (positions[Position(idx=back_g_row[g_idx], len_g=len(back_g_row), len_e = len(back_e_row))][back_e_row[a_idx]] + beta) * (g_e_counts[g][e] + theta) * 1. / (g_counts[g] + theta * e_vocab_size )
            prob_table.append(prob_g)
        sampled = sample_from(prob_table) - 1
        alignment[a_idx] = sampled
    for e_idx, (g_idx,e) in enumerate(zip(alignment, e_sen)):
        increment_count(g_idx, e_idx, e, 1)
    return


def create_vocab(sentences, split_dict):
    all_elems = dict()
    for w in split_dict:
        if w not in all_elems:
            all_elems[w] = len(all_elems)
        for s in split_dict[w]:
            if s not in all_elems:
                all_elems[s] = len(all_elems)
    to_return = []
    expanded_sens = []
    map_backs = []
    for s in sentences:
        to_append = []
        expanded = []
        map_back = []
        for ind, w in enumerate(s):
            to_append.append(all_elems[w])
            for sp in split_dict[w]:
                expanded.append(all_elems[sp])
                map_back.append(ind)
        expanded_sens.append(expanded)
        map_backs.append(map_back)
        to_return.append(to_append)
    return all_elems, to_return, expanded_sens, map_backs

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

def record(alignments, recorded, back_e, back_f):
    # print 'recording'
    assert len(alignments) == len(recorded)
    for i in range(len(recorded)):
        assert len(alignments[i]) == len(recorded[i])
        for j in range(len(alignments[i])):
            al = alignments[i][j]
            if al == -1:
                recorded[i][back_e[i][j]].update([-1])
            else:
                recorded[i][back_e[i][j]].update([back_f[i][alignments[i][j]]])
    return

def output_record(recorded, epoch, great_epoch, theta, beta, prefix):
    import cPickle
    with open(u'{}_epoch_{}_great_epoch_{}_theta_{}_beta_{}'.format(prefix, epoch, great_epoch, theta, beta), mode='w') as fh:
        for record in recorded:
            fh.write(u' '.join([unicode(x.most_common(1)[0][0]) for x in record])+u'\n')
    with open(u'{}_epoch_{}_great_epoch_{}_theta_{}_beta_{}.save'.format(prefix, epoch, great_epoch, theta, beta), mode='wb') as fh:
        cPickle.dump(recorded, fh, protocol=cPickle.HIGHEST_PROTOCOL)

def load_split(fname):
    if fname is None:
        return dict()
    orig_name = fname
    split_name = fname + '-split'
    to_return = dict()
    with open(orig_name, mode='r', encoding='utf-8') as orig_fh, open(split_name, mode='r', encoding='utf-8') as split_fh:
        orig_lines = [x.strip() for x in orig_fh]
        split_lines = [x.strip() for x in split_fh]
    for orig, split in zip(orig_lines, split_lines):
        to_return[orig] = set(split.split())
    return to_return

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('aligned')
    parser.add_argument('--output-prefix', default='output')
    parser.add_argument('--split-f', default=None)
    parser.add_argument('--split-e', default=None)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    if not args.reverse:
        split_f_words = load_split(args.split_f)
        split_e_words = load_split(args.split_e)
        (f,e) = read_aligned(args.aligned)
        f_vocab, numbered_f, expanded_f, back_f = create_vocab(f, split_f_words)
        e_vocab, numbered_e, expanded_e, back_e = create_vocab(e, split_e_words)
    else:
        split_f_words = load_split(args.split_f)
        split_e_words = load_split(args.split_e)
        (f,e) = read_aligned(args.aligned)
        e_vocab, numbered_e, expanded_e, back_e = create_vocab(f, split_f_words)
        f_vocab, numbered_f, expanded_f, back_f = create_vocab(e, split_e_words)

    # print len(f_vocab), len(e_vocab)

    test_record = True

    rec = None

    # great epochs
    num_great_epochs = 100

    # epochs
    num_epochs = 100
    
    burnins = 10

    record_every = 10

    theta = 1e-5
    beta = 1.

    for great_epoch in range(num_great_epochs):
        (alignments, f_counts, f_e_counts, positions) = init_align(expanded_e,expanded_f, back_e, back_f)
        if rec is None:
            rec = init_record(back_e)
        for epoch in range(num_epochs):
            print 'epoch {}:'.format(epoch)
            shuffled = range(len(alignments))
            random.shuffle(shuffled)
            for sen_idx in shuffled:
                sample(alignments[sen_idx], expanded_f[sen_idx], expanded_e[sen_idx], f_counts, f_e_counts, positions, len(e_vocab), back_f[sen_idx], back_e[sen_idx], theta, beta)
            if epoch + 1 > burnins:
                record(alignments, rec, back_e, back_f)
                if epoch % record_every == 0 or test_record is True:
                    output_record(rec, epoch, great_epoch, theta, beta, args.output_prefix)
    return

if __name__ == '__main__':
    main()
