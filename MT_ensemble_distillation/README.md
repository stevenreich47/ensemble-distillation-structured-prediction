This repo contains modifications to Fairseq to support ensemble
distillation, as described in:

Ensemble Distillation for Structured Prediction: Calibrated, Accurate,
Fast - Choose Three. EMNLP (2020).
Steven Reich, David Mueller, Nicholas Andrews
https://arxiv.org/abs/2010.06721

If you use this code in published work, please cite the paper above.

The official Fairseq repo is here: https://github.com/pytorch/fairseq
Many thanks to the developers!

# Environment

This is based on Fairseq 0.9.0. Please see `conda_list.txt` for a list
of a working conda virtual environment.

*Note*: the changes are fairly self-contained and it should be
 possible to apply them to the latest verion of Fairseq without too
 much trouble.

# Experiment scripts

See `experiments/de2en_ens_dist` for experiment scripts for the
`de2en` task. You can do the other direction by swapping the `src` and
`tgt` arguments in these scripts.

The steps are as follows using the scripts in that directory:

1. `preprocess.sh`: Preprocess the data.

2. `fit_teacher.sh`: Train an ensemble by fitting `N` separate models with different
random seeds. We use the standard recipes provided by Fairseq for this
for Transformers.

3. `cache_ensemble_predictions.sh`: Memoize the predictive distribution of the ensemble.

4. `distill_ensemble.sh`: Distill the ensemble into a single model using the cached probs.

5. `eval.sh`: Evaluation script.
