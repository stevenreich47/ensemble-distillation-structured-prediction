This repo contains code to support ensemble distillation for Named Entity 
Recognition, as originally described in the EMNLP 2020 paper:

Ensemble Distillation for Structured Prediction: Calibrated, Accurate,
Fast - Choose Three  
Steven Reich, David Mueller, Nicholas Andrews  
https://arxiv.org/abs/2010.06721  

If you use this code in published work, please cite the paper above.

# Environment

Create and activate a dedicated virtual environment
``` bash
conda create -n ner pip python=3.7
conda activate ner
```

Install necessary Python dependencies
```bash
pip install -r requirements.txt
```

Configure for development
```bash
python setup.py develop
```

Define the following environment variables, e.g. in `.bashrc`:

* `NER_REPO_DIR`: directory containing this README
* `NER_EXP_DIR`: where to save model files, etc.
* `NER_DATA_DIR`: location of iob2 processed CoNLL 2003 splits 
* `MBERT_DIR`: directory containing vocab and ckpt for m-BERT

# Experiment scripts

`experiments/conll_de` contains scripts for the CoNLL 2003 German NER task.
Experiments for the English task should require only minor modifications.

The steps are as follows using the scripts in that directory:

1. `make_tfrecords.sh`: Convert text files of CoNLL data into tfrecords files.

2. `train_teacher.sh MODEL_TYPE TEACHER_NUM`: Train a single model and get predictions on each data split. `MODEL_TYPE` can be either "iid" or "crf". `TEACHER_NUM` is an integer between 0 and 8 to distinguish the model (and determine what piece of the training data is held out for validation). Before proceeding to the next step, you must run this for each value of `TEACHER_NUM` for your chosen `MODEL_TYPE`.

3. `python combine_dists.py MODEL_TYPE`: Combines the train-set predictions of all 9 teacher models into one file for use by the next script.

4. `teacher_tfrecords.sh MODEL_TYPE`: Writes tfrecords which include the teacher distributions to be used as targets for the student model.

5. `distill_ensemble.sh MODEL_TYPE NUM_TEACHERS`: Trains a single student model distilled from an ensemble of `NUM_TEACHERS` models of type `MODEL_TYPE`.

6. `python measure_f1.py MODEL_TYPE SPLIT`: Outputs F1 scores for all `MODEL_TYPE` teachers and students on `SPLIT` data.

7. `python measure_calibration.py MODEL_TYPE SPLIT`: Outputs calibration metrics for all `MODEL_TYPE` teachers and students on `SPLIT` data.
