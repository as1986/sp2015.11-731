#!/usr/bin/python
from io import open

def main():
    import sys
    with open(sys.argv[1], encoding='utf-8', mode='r') as fh:
        for l in fh:
            line_output = []
            for idx, w in enumerate(l.strip().split()):
                line_output.append(u'{}-{}'.format(w,idx))
            print u' '.join(line_output)

if __name__ == '__main__':
    main()
