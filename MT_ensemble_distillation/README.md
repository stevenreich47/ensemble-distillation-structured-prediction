This repo contains modifications to [Fairseq](https://github.com/pytorch/fairseq) to support ensemble
distillation, as originally described in the EMNLP 2020 paper:

Ensemble Distillation for Structured Prediction: Calibrated, Accurate,
Fast - Choose Three  
Steven Reich, David Mueller, Nicholas Andrews  
https://arxiv.org/abs/2010.06721  

If you use this code in published work, please cite the paper above.
The Fairseq library is described in the paper below:

Ott, Myle, et al. "fairseq: A fast, extensible toolkit for sequence
modeling." arXiv preprint arXiv:1904.01038 (2019).

Many thanks to the Fairseq developers for a great library!

# Environment

This is based on Fairseq 0.9.0 so your environment should be compatible
with that. The Fairseq README provides instructions for installing the required packages.
Please see `conda_list.txt` for a list of a working conda virtual environment
that was used to run experiments reported in the paper above.

*Note*: the changes are fairly self-contained and it should be
 possible to apply them to the latest verion of Fairseq without too
 much trouble. Specifically:

* `fairseq_cli/validate.py` was modified to support caching a (truncated) predictive distribution to an hdf5 file
* `fairseq/criterions/distillation_cross_entropy.py` is the implementation of the distillation objective
* `fairseq/data/language_pair_dataset_with_teacher.py` and some associated files were modified to read and pass around the teacher predictive distribution

# Experiment scripts

See `experiments/de2en_ens_dist` for experiment scripts for the
`de2en` task. You can do the other direction by swapping the `--source-lang` and
`--target-lang` arguments in these scripts.

Set the following environment variables:
`RAW_DATA` - path to raw WMT data
`DATA_DIR` - where to output the preprocessed data
`JOBS_DIR` - where to save directories of model checkpoints
`OUTPUT_DIR` - where to save .h5 files of (truncated) teacher distributions 

The steps are as follows using the scripts in that directory:

1. `preprocess.sh`: Preprocess the data.

2. `fit_teacher.sh`: Train an ensemble by running this script `N` times with different
random seeds. We use the standard recipes provided by Fairseq for this
for Transformers.

3. `cache_ensemble_predictions.sh`: Memoize the predictive distribution of the ensemble. For example, to cache the top-64 predictions from three models called `ce1`, `ce2`, and `ce3`, you would run: `./cache_ensemble_predictions.sh 64 last ce1 ce2 ce3`. This will produce an `h5` file containing the cached probabilities.

4. `distill_ensemble.sh`: Distill the ensemble into a single model using the cached probs. For example: `./distill_ensemble.sh de2en_ens3_64 64 1 0.5 300000 kl 0.1 0.0007 0.1 ce1_ce2_ce3_64_last.h5`.

5. `eval.sh`: Evaluation script (BLEU score). Takes as an argument the directory within `$JOBS_DIR` containing the checkpoints for the model to be evaluated. For example: `./eval.sh ce1`.

6. `calibration.sh`: Evaluate calibration. Takes as an argument a checkpoint within `$JOBS_DIR` to be evaluated (if multiple checkpoints are passed, ensemble predictions are evaluated). For example: `./calibration.sh ce1/checkpoint_last.pt ce2/checkpoint_last.pt ce3/checkpoint_last.pt`.
