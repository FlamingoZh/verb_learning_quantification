import json
import re
import sys
from collections import Counter

# base_path = '/user_data/yuchenz2/raw_data_verb_alignment/vrd/sg_dataset/'
base_path = '/Users/yuchen/Downloads/sg_dataset/'
annotations = json.load(open(base_path+"sg_train_annotations.json", "rb"))

nouns_syn=list()

for image in annotations:
	if not image["relationships"]:
		continue

	objects=[(item['names'][0],(item['bbox'])) for item in image['objects']]

	for relation in image['relationships']:
		if not relation['text'] or not relation['objects'] or not relation['relationship']:
			continue

		for obj_idx in relation['objects']:
			obj_name=objects[obj_idx][0]
			
			# remove numbers in object names (e.g., 'car 2')
			obj_name_without_number=list()
			for item in obj_name.split():
				if not item.isdigit():
					obj_name_without_number.append(item)
			obj_name_without_number=" ".join(obj_name_without_number)
			
			nouns_syn.append(obj_name_without_number)

sorted_dict = sorted(dict(Counter(nouns_syn)).items(), key=lambda x: x[1], reverse=True)

with open('../data/concepts/vrd_noun_count_all.txt',"w") as f:
	for i in sorted_dict:
		f.write(str(i[0])+" "+str(i[1])+"\n")

with open('../data/concepts/vrd_noun_concept_least20.txt',"w") as f:
	for i in sorted_dict[:75]:
		f.write(str(i[0])+"\n")

with open('../data/concepts/vrd_noun_concept_least10.txt',"w") as f:
	for i in sorted_dict[:109]:
		f.write(str(i[0])+"\n")