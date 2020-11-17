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


""" Utility function to get string list of available GPUs
Return estimator configurations given num_gpu and checkpoint steps
"""

import os

import tensorflow.compat.v1 as tf
from absl import logging
from tensorflow.python.client import device_lib


def get_run_config(num_gpu=0, num_cpu=6, save_checkpoints_steps=10_000,
                   num_checkpoints=1, keep_checkpoint_hours=10_000) \
    -> tf.estimator.RunConfig:
  """Create Estimator configs differentiating between standard or
  MirroredStrategy

  Arguments:
    :param num_gpu: Number of GPU to use.
    :param num_cpu: Number of CPU to use.
    :param save_checkpoints_steps: Steps between saving model checkpoints.
    :param num_checkpoints: Number of checkpoints to save.
    :param keep_checkpoint_hours:

  """

  if num_gpu > 1:
    gpus = get_available_gpus()[:num_gpu]
    if not gpus:
      raise Exception('No GPU available.')
    if len(gpus) != num_gpu:
      raise Exception(f'Requested {num_gpu} GPU, only {len(gpus)} available')
    strategy = tf.distribute.MirroredStrategy(devices=gpus)
  else:
    strategy = None

  if num_gpu:
    cuda_env = os.environ["CUDA_VISIBLE_DEVICES"]
    logging.info(f"CUDA_VISIBLE_DEVICES: {cuda_env}")
    device_list = device_lib.list_local_devices()
    logging.info(f"TF devices: {device_list}")

  sess_config = tf.ConfigProto(
      intra_op_parallelism_threads=num_cpu,
      inter_op_parallelism_threads=num_cpu,
      allow_soft_placement=True)

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=save_checkpoints_steps,
      save_checkpoints_secs=None,
      keep_checkpoint_max=num_checkpoints,
      keep_checkpoint_every_n_hours=keep_checkpoint_hours,
      session_config=sess_config,
      log_step_count_steps=100,
      save_summary_steps=100,
      train_distribute=strategy)

  return run_config


def get_available_gpus():
  # to avoid allocating all GPU memory, take a small chunk of it first
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.Session(config=config)
  local_device_protos = device_lib.list_local_devices()
  # get names of all available GPU
  return [x.name for x in local_device_protos if x.device_type == 'GPU']
