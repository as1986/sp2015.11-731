#!/usr/bin/python

from io import open

def load_alignments(fname, reverse=False):
    to_return = []
    with open(fname, mode='r', encoding='utf-8') as fh:
        for line in fh:
            to_add = set()
            each_line = line.strip().split()
            for al in each_line:
                if not reverse:
                    (begin, end) = al.split('-')
                else:
                    (end, begin) = al.split('-')
                to_add.add((int(begin), int(end)))
            to_return.append(to_add)
    return to_return

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('forward')
    parser.add_argument('reverse')
    args = parser.parse_args()
    (fe, ef) = load_alignments(args.forward), load_alignments(args.reverse, reverse=True)

    def is_neighbor(cand, aligned):
        if abs(cand[0] - aligned[0]) < 2 and abs(cand[1] - aligned[1]) < 2:
            return True
        else:
            return False

    for ef_line, fe_line in zip(ef, fe):
        intersection = ef_line & fe_line
        union = ef_line | fe_line
        aligned = {x for x in intersection}
        aligned_e = {x[1] for x in aligned}
        aligned_f = {x[0] for x in aligned}
        while True:
            to_add = set()
            for each_aligned in aligned:
                for candidate in union:
                    if (candidate[0] not in aligned_f or candidate[1] not in aligned_e) and is_neighbor(candidate, each_aligned):
                        to_add.add(candidate)
                        aligned_e.add(candidate[1])
                        aligned_f.add(candidate[0])
            if len(to_add)==0:
                break
            aligned |= to_add
        # final ef
#        for each_ef in ef_line:
#            if(each_ef[0] not in aligned_f or each_ef[1] not in aligned_e):
#                aligned.add(each_ef)
#                aligned_e.add(each_ef[1])
#                aligned_f.add(each_ef[0])

        # final fe
#        for each_fe in fe_line:
#            if(each_ef[0] not in aligned_f or each_ef[1] not in aligned_e):
#                aligned.add(each_ef)
#                aligned_e.add(each_ef[1])
#                aligned_f.add(each_ef[0])
        to_print = [u'{}-{}'.format(x[0], x[1]) for x in sorted(aligned, key=lambda x: x[0])]
        print u' '.join(to_print)
if __name__ == '__main__':
    main()
