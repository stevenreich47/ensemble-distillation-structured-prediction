#! /usr/bin/env python

# Copyright 2019 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""
Train NER model from TFRecord files. The TFRecord inputs should
follow a `DataFormat` specified in Registries.data_formats.
"""

import argparse as ap
import logging as py_logging
import os

from absl import logging

from griffin.trainer import Trainer

# From https://stackoverflow.com/questions/35911252
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

py_logging.getLogger("tensorflow").setLevel(py_logging.INFO)


def get_arguments():
  """ Parses the arguments """
  p = ap.ArgumentParser()
  p.add_argument('mode', type=str, choices=['fit', 'predict'])
  p.add_argument('--no-sanity-checks', default=False, action='store_true',
                 help="Disable sanity checks on input tfrecords")
  p.add_argument('--eval-throttle-secs', default=60, type=int)
  p.add_argument('--train-tfrecord-path', default=None, type=str)
  p.add_argument('--dev-tfrecord-path', default=None, type=str)
  p.add_argument('--hold-out-fraction', default=None, type=float)
  p.add_argument('--test-tfrecord-path', default=None, type=str)
  p.add_argument('--data-format', required=True, type=str)
  p.add_argument('--model', default='feature_bi_crf_model', type=str)
  p.add_argument('--hparams', default='bi_crf_default_9', type=str)
  p.add_argument('--hparams-str', type=str,
                 help="Update `hparams` from comma separated list "
                      "of name=value pairs")
  p.add_argument('--hparams-json', type=str,
                 help="Update `hparams` from parameters in JSON file")
  p.add_argument('--label-map-path', type=str, default='label_map.pickle')
  p.add_argument('--model-path', type=str, default='checkpoints')
  p.add_argument('--warm-start-from', type=str)
  p.add_argument('--warm-start-vars', type=str, default='*')
  p.add_argument('--output-file', type=str, default='tags.txt')
  p.add_argument('--output-confidences', type=str, default='confidences.txt')
  p.add_argument('--conf-method', type=str, default='average',
                 help="max, min, average across entity spans")
  p.add_argument('--report-confidences', type=bool, default=False)
  p.add_argument('--full-dist', type=bool, default=False)
  p.add_argument('--bio-dist', type=bool, default=False)
  p.add_argument('--logit-dist', type=bool, default=False)
  p.add_argument('--temperature', type=float, default=None)
  p.add_argument('--save-checkpoints-steps', type=int, default=10_000)
  p.add_argument('--keep-checkpoint-hours', type=int, default=10_000)
  p.add_argument('--train-epochs', default=100, type=int)
  p.add_argument('--repeat', default=1, type=int,
                 help="How many repetitions of the data constitutes an epoch")
  p.add_argument('--min-epochs', default=2, type=int)
  p.add_argument('--early-stop-patience', default=3, type=int)
  p.add_argument('--train-batch-size', type=int, default=64)
  p.add_argument('--eval-batch-size', type=int, default=64)
  p.add_argument('--shuffle-buffer-size', type=int, default=1_000)
  p.add_argument('--random-seed', type=int, default=None)
  p.add_argument('--n-gpu', type=int, default=0)
  p.add_argument('--n-cpu', type=int, default=6)
  p.add_argument('--token-calibrate', type=bool, default=False)
  p.add_argument('--logging-names', type=str, default="learning_rate")
  p.add_argument('--sliding-window-length', type=int, default=-1)
  p.add_argument('--sliding-window-context', type=int, default=-1)
  p.add_argument('--custom-split', type=int, default=None)
  p.add_argument('--which-model', type=int, default=None)
  return p.parse_args()


def main(args):
  # Disable TF2 behavior globally
  # tf.compat.v1.disable_v2_behavior()

  # Logging
  logging.set_verbosity(logging.INFO)

  trainer = Trainer(
      sanity_checks=not args.no_sanity_checks,
      eval_throttle_secs=args.eval_throttle_secs,
      train_tfrecord_path=args.train_tfrecord_path,
      hold_out_fraction=args.hold_out_fraction,
      test_tfrecord_path=args.test_tfrecord_path,
      data_format=args.data_format,
      model=args.model,
      hparams=args.hparams,
      hparams_str=args.hparams_str,
      hparams_json=args.hparams_json,
      label_map_path=args.label_map_path,
      model_path=args.model_path,
      warm_start_from=args.warm_start_from,
      warm_start_vars=args.warm_start_vars,
      output_file=args.output_file,
      output_confidences=args.output_confidences,
      conf_method=args.conf_method,
      report_confidences=args.report_confidences,
      full_dist=args.full_dist,
      bio_dist=args.bio_dist,
      logit_dist=args.logit_dist,
      temperature=args.temperature,
      save_checkpoints_steps=args.save_checkpoints_steps,
      keep_checkpoint_hours=args.keep_checkpoint_hours,
      train_epochs=args.train_epochs,
      repeat=args.repeat,
      min_epochs=args.min_epochs,
      early_stop_patience=args.early_stop_patience,
      train_batch_size=args.train_batch_size,
      eval_batch_size=args.eval_batch_size,
      shuffle_buffer_size=args.shuffle_buffer_size,
      random_seed=args.random_seed,
      n_cpu=args.n_cpu,
      n_gpu=args.n_gpu,
      token_calibrate=args.token_calibrate,
      logging_names=args.logging_names,
      dev_tfrecord_path=args.dev_tfrecord_path,
      sliding_window_length=args.sliding_window_length,
      sliding_window_context=args.sliding_window_context,
      custom_split=args.custom_split,
      which_model=args.which_model
  )

  if args.mode == "fit":
    trainer.fit()

  elif args.mode == "predict":
    trainer.predict()

  else:
    raise ValueError(args.mode)


if __name__ == '__main__':
  main(get_arguments())
