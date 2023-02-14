import json
import re
import sys
from collections import Counter

# base_path = '/user_data/yuchenz2/raw_data_verb_alignment/vrd/sg_dataset/'
base_path = '/Users/yuchen/Downloads/sg_dataset/'
annotations = json.load(open(base_path+"sg_train_annotations.json", "rb"))

verbs_syn=list()

for image in annotations:
	if not image["relationships"]:
		continue

	for relation in image['relationships']:
		if not relation['text'] or not relation['objects'] or not relation['relationship']:
			continue

		verbs_syn.append(relation['relationship'])

sorted_dict = sorted(dict(Counter(verbs_syn)).items(), key=lambda x: x[1], reverse=True)

with open('../data/concepts/vrd_verb_count_all.txt',"w") as f:
	for i in sorted_dict:
		f.write(str(i[0])+" "+str(i[1])+"\n")

with open('../data/concepts/vrd_verb_concept_least20.txt',"w") as f:
	for i in sorted_dict[:75]:
		f.write(str(i[0])+"\n")

with open('../data/concepts/vrd_verb_concept_least10.txt',"w") as f:
	for i in sorted_dict[:109]:
		f.write(str(i[0])+"\n")