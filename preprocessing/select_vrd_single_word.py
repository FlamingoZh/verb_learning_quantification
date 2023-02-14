# select concepts that are single words and appear at least 20 times in the dataset

def select_single_word(input_concept_file,output_concept_file,num_concept):
	selected_concepts=list()
	with open(input_concept_file,"r") as f:
		lines = f.readlines()
		for line in lines:
			concept=line.strip()
			if " " not in concept:
				selected_concepts.append(concept)

	with open(output_concept_file,"w") as f:
		for concept in selected_concepts[:num_concept]:
			f.write(concept+"\n")


select_single_word('../data/concepts/vrd_noun_concept_least20.txt','../data/concepts/vrd_noun_concept_least20_single_word.txt',40)
select_single_word('../data/concepts/vrd_noun_concept_least10.txt','../data/concepts/vrd_noun_concept_least10_single_word.txt',54)
select_single_word('../data/concepts/vrd_verb_concept_least20.txt','../data/concepts/vrd_verb_concept_least20_single_word.txt',40)
select_single_word('../data/concepts/vrd_verb_concept_least10.txt','../data/concepts/vrd_verb_concept_least10_single_word.txt',54)

