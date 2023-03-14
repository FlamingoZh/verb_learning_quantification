import pickle
import numpy as np
import pprint
import random
import sys
import argparse

from permutation import permutation
from utils.utils_funcs import sample_embeddings_visual, sample_embeddings_language

def simulate_exemplar_aggregation(datasetname, data, aggregation_mode, n_exemplar_max=20, n_sample=1, extra_info=None):

    mean_list = [0]
    max_list = [0]
    min_list = [0]
    n_exemplar_list = [0]
    y_list= [0]

    print("Start simulation...")
    for sample in range(1, n_sample + 1):
        print(f"Sample {sample}:")

        if aggregation_mode=="visual":
            sampled_data=sample_embeddings_visual(data,n_exemplar_max)
        elif aggregation_mode=="language":
            sampled_data=sample_embeddings_language(data,n_exemplar_max)
        else:
            print("Error, unrecognized aggregation mode.")
            sys.exit(1)

        exemplar_all=list()
        exemplar_indices=np.arange(n_exemplar_max)
        exemplar_all.append(exemplar_indices)
        for i in range(n_exemplar_max,1,-1):
            exemplar_indices=np.random.choice(exemplar_indices,len(exemplar_indices)-1, replace=False)
            exemplar_all.append(exemplar_indices)

        for n_exemplar in range(n_exemplar_max,0,-1):
            print(f"n_exemplar {n_exemplar}")

            #compute alignment strength
            words=sampled_data['words']
            visual_agg=list()
            lang_agg=list()
            # aggregate embeddings
            for word in words:
                if aggregation_mode=="visual":
                    visual_agg.append(np.mean(sampled_data['embeds'][word]['visual'][exemplar_all[n_exemplar_max-n_exemplar]],axis=0))
                    lang_agg.append(sampled_data['embeds'][word]['language'])
                elif aggregation_mode=="language":
                    visual_agg.append(sampled_data['embeds'][word]['visual'])
                    lang_agg.append(np.mean(sampled_data['embeds'][word]['language'][exemplar_all[n_exemplar_max-n_exemplar]],axis=0))

            z_0 = np.stack(visual_agg)
            z_1 = np.stack(lang_agg)

            relative_alignment_strength, alignment_strength_list = permutation(z_0, z_1)
            n_exemplar_list.append(n_exemplar)
            y_list.append(relative_alignment_strength)
            print("Relative Alignment: {}".format(relative_alignment_strength))

    plot_data = dict(
        n_exemplar_list=n_exemplar_list,
        y_list=y_list,
        datasetname=datasetname,
        n_exemplar_max=n_exemplar_max,
        n_sample=n_sample,
        aggregation_mode=aggregation_mode,
        extra_info=extra_info
    )

    if extra_info:
        file_name = "_".join([datasetname, aggregation_mode, str(n_exemplar_max), str(n_sample), extra_info])
    else:
        file_name = "_".join([datasetname, aggregation_mode, str(n_exemplar_max), str(n_sample)])

    pickle.dump(plot_data, open('../data/dumped_plot_data/' + file_name + '.pkl', 'wb'))
    print("\n data for plotting are dumped to /data/dumped_plot_data/" + file_name + ".pkl")

if __name__ == '__main__':

    print("aggregate_exemplars_subsample.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("datasetname", help="name of the dataset")
    parser.add_argument("data", help="data to read from")
    parser.add_argument("aggregation_mode", help="aggregation mode: language, visual, language_visual")
    parser.add_argument("--n_exemplar_max", type=int, default=20, help="maximum number of exemplars in simulation")
    parser.add_argument("--n_sample", type=int, default=1, help="number of samples in each run")
    parser.add_argument("--extra_info", default=None, help="extra information")
    args = parser.parse_args()

    data = pickle.load(open(args.data, 'rb'))
    simulate_exemplar_aggregation(args.datasetname,
                                  data,
                                  args.aggregation_mode,
                                  n_exemplar_max=args.n_exemplar_max,
                                  n_sample=args.n_sample,
                                  extra_info=args.extra_info
                                  )
