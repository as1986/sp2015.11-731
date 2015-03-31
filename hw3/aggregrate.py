#!/usr/bin/python
from io import open

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    likelihoods = None
    lines = None

    for f in args.files:
        prob_file = f + '.prob'
        with open(f, mode='r', encoding='utf-8') as fh, open(prob_file, mode='r', encoding='utf-8') as prob_fh:
            this_lines = [x for x in fh]
            this_ll = [float(x.strip()) for x in prob_fh]
            if likelihoods is None:
                likelihoods = this_ll
                lines = this_lines
            else:
                for i, likelihood in enumerate(this_ll):
                    if likelihoods[i] < likelihood:
                        sys.stderr.write('old: {} new: {}'.format(likelihoods[i], likelihood))
                        likelihoods[i] = likelihood
                        lines[i] = this_lines[i]
    o_fh = open(args.outfile, mode='w', encoding='utf-8')
    o_fh.writelines(lines)
    o_fh.close()

    return

if __name__ == '__main__':
    main()
