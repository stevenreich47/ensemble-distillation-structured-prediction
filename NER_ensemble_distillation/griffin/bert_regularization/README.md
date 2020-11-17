# BERT L2 regularization

This is an implementation of a variant of L2 regularization which can be applied to a select 
number of layers in a model. When provided with an initial checkpoint and a regex of the layers to 
regularize, this method regularizes the specified weights towards the values provided in the checkpoint.

To use this regularization method, be sure to set `use_l2` to `True` when adding BERT hyperparameters to your hparams.
This adds 3 new hparams: `l2_vars`, the regex denoting the variables to include in the l2 penalty, `lam`, the weight
of the l2 penalty, and `init_checkpoint`, the path to the checkpoint containing the weight values to regularize towards.
Each of the default values of these hyperparameters can be overwritten in `train_from_features.py` using `hparams-str` 
to have the appropriate values for your task. An example of an appropriate set of hyperparameters is 
`bert_rubert_bi_crf_adafactor_clipped_aligned_l2`, which can be found in `ner/models/bert.py`. 

NOTE: `run_expt.sh` assumes that you have already run `experiments/jhu/bert_embed/ru_with_alignment/make_tfrecords.sh`
