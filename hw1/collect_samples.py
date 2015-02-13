#!/usr/bin/python

def main():
   import cPickle, sys
   to_combine = sys.argv[2:-1]
   with open(sys.argv[1], 'rb') as base_fh:
      combined = cPickle.load(base_fh)
   for fn in to_combine:
      with open(fn, 'rb') as fh:
         tmp = cPickle.load(fh)
         assert len(tmp) == len(combined)
         for tmp_l, combined_l in zip(tmp, combined):
            for tmp_a, combined_a in zip(tmp_l, combined_l):
               combined_a.update(tmp_a)

   model1_new = __import__('model1-new')
   model1_new.output_record(combined, 'UNK', 'UNK', 'UNK', 'UNK', sys.argv[-1])
   return

if __name__ == '__main__':
   main()
