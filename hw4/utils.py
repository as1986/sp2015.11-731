import sys
from collections import defaultdict

def read_ttable(filename):
	translation_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
	print >>sys.stderr, 'Reading ttable from %s...' % filename
	with open(filename) as f:
		for i, line in enumerate(f):
			source, target, features = [part.strip() for part in line.decode('utf-8').strip().split('|||')]
			features = [float(v) for v in features.split()]
			assert len(features) == 4
			features = { 'log_prob_tgs': features[0], \
				     'log_prob_sgt': features[1], \
				     'log_lex_prob_tgs': features[2], \
				     'log_lex_prob_sgt': features[3] }
			translation_table[source][target] = features
			sys.stderr.write('%d\r' % i)
	print >>sys.stderr
	return translation_table

class DependencyTree:
	def __init__(self, n):
		self.terminals = [None for _ in range(n)]
		self.tags = [None for _ in range(n)]
		self.children = [[] for _ in range(n)]
		self.parents = [None for _ in range(n)]
		self.roots = []

	@staticmethod
	def parse(input_string):
		input_lines = input_string.strip().split('\n')
		n = len(input_lines)
		tree = DependencyTree(n)
		for i, line in enumerate(input_lines):
			fields = [field.strip() for field in line.strip().split('\t')]
			j, terminal, _, tag, __, ___, parent, relation = fields
			j = int(j)
			parent = int(parent)
			assert i + 1 == j
			assert 1 <= j <= n
			assert 0 <= parent <= n
			tree.terminals[i] = terminal
			tree.tags[i] = tag
			if parent != 0:
				tree.parents[i] = (parent - 1, relation)
				tree.children[parent - 1].append((i, relation))
			else:
				tree.parents[i] = (None, relation)
				tree.roots.append((i, relation))

		return tree

def read_dep_trees(filename):
	current_tree = []
	with open(filename) as f:
		for line in f:
			line = line.decode('utf-8')
			if len(line.strip()) != 0:
				current_tree.append(line.strip())
			else:
				yield DependencyTree.parse('\n'.join(current_tree))
				current_tree = []
