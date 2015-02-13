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
                to_add.add((begin, end))
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
        if abs(cand[0] - aligned[0]) < 1 and abs(cand[1] - aligned[1]) < 1:
            return True
        else:
            return False

    for ef_line, fe_line in zip(ef, fe):
        intersection = ef_line & fe_line
        union = ef_line | fe_line
        aligned = {x for x in intersection}
        aligned_e = {x[1] for x in aligned}
        aligned_f = {x[0] for x in aligned}
        add_flag = False
        while True:
            for each_aligned in aligned:
                for candidate in union:
                    if (candidate[0] not in aligned_f or candidate[1] not in aligned_e) and candidate in union and is_neighbor(candidate, each_aligned):
                        aligned.add(candidate)
                        aligned_e.add(candidate[1])
                        aligned_f.add(candidate[0])
                        add_flag = True
            if add_flag == False:
                break
        print aligned
                

if __name__ == '__main__':
    main()
