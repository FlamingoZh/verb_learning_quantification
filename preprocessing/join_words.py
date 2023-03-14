import os
import sys
import numpy as np

def join_words(file):
	"""
	read words from a file, join them with comma and space
	"""
	with open(file,"r") as f:
		words=list()
		for line in f.readlines():
			# print(line.strip().split(".")[0])
			words.append(line.strip().split(".")[0])
	return ", ".join(words)


if __name__ == '__main__':
	# joined_words=join_words("../data/concepts/vg_noun_concept_least20.txt")
	# joined_words=join_words("../data/concepts/vg_verb_concept_least20.txt")
	joined_words=join_words("../data/concepts/mit_concept.txt")
	# joined_words=join_words("../data/concepts/vrd_noun_concept_least10_single_word.txt")
	# joined_words=join_words("../data/concepts/vrd_verb_concept_least10_single_word.txt")
	print(joined_words)