# Verb Learning Quantification

This repository provides implementations of analyses in the paper Quantifying the Roles of Visual, Linguistic, and Visual-Linguistic Complexity in Verb Acquisition.

## Getting Started

### Clone the repository from github

```
git clone git@github.com:FlamingoZh/verb_learning_quantification.git
```

### Requirements

- Python >= 3.8
- Other dependencies: numpy, scipy, scikit-learn, pandas, pytorch, torchvision, transformers, jupyterlab, matplotlib, seaborn, statsmodels, opencv-python, pytest-shutil, xgboost, shap

### Datasets

We conducted anaylses on [Visual Genome](http://visualgenome.org/), [Visual Relationship Detection](https://cs.stanford.edu/people/ranjaykrishna/vrd/), and [Moments in Time](http://moments.csail.mit.edu/) dataset. After downloading the data, make sure that you specify the path to the dataset by modifying `base_path` in `python/utils/data_generation_library.py`.

### Pretrained Models

The vision model we used is an unsupervised model with ResNet-50 architecture called Swapping Assignments between Views (SwAV). You can find several pretrained models from their [homepage](https://github.com/facebookresearch/swav). In our analyses, we used the one trained with 800 epochs. After downloading the pretrained model, put it under `pretrained_models/` (if you don't have this folder, create it).

We also employed the uncased Bidirectional Encoder Representations from Transformers (BERT) model from the [Hugging face Transformers library](https://huggingface.co/docs/transformers/model_doc/bert) as the language model. The model should be downloaded automatically the first time you run the script for sampling exemplars.

## Sample Learning Exemplars

The first thing to do is to generate samples of visual and language representations of words and store them on disk for faster future computation. An example is as follows:

```
python python/gen_data.py vg_noun vg_noun_concept_least20.txt bert swav --n_sample 20 --cuda
```

The embeddings will be stored in `data/dumped_embeddings/`.

## Aggregate Exemplars

One-dimensional (visual or lingusitic) aggregation is achieved by `python/aggregate_exemplars.py`. An example is as follows:

```
python python/aggregate_exemplars.py vg_noun "../data/dumped_embeddings/vg_noun_least20_swav_bert_20.pkl" visual \
  --n_exemplar_max 10 \
  --n_sample 1000
```

This script will create a pickle file in `data/dumped_plot_data/` so you that you can load the pickle file in `notebooks/1D_exemplar_aggregation.ipynb` to make plots.

Two-dimensional aggregation is achieved by `python/aggregate_exemplars_2D.py`. An example is as follows:

```
python python/aggregate_exemplars_2D.py vg_noun "../data/dumped_embeddings/vg_noun_least20_swav_bert_20.pkl" visual_language \
  --n_l_exemplar_max 8 \
  --n_v_exemplar_max 8 \
  --n_sample 500
```

Similarly, this script will create a pickle file in `data/dumped_plot_data/` and you can make plots by running `notebooks/2D_aggregation_VG.ipynb`.

## Regression Analysis

Word frequency from CHILDES and Age of Acquisition from Wordbank can be found in `data/processed` (you can also generate them on your own by running `R/get_freq_and_aoa_vg.rmd`). Code for regression analysis can be found in `notebooks/xgboost_VG.ipynb`.
