""" Module for restoring parameter values from a checkpont """

import collections

import tensorflow.compat.v1 as tf


def init_vars_from_checkpoint(ckpt_to_initialize_from, new_scope, l2_vars):
  """ Args: checkpoint_to_initialize_from: checkpoint from which parameter
  values
      will be restored.
      new_scope: scope for corresponding variables in new model. """
  model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, new_scope)
  model_vars_by_name = {}
  for model_var in model_vars:
    name = model_var.name
    model_vars_by_name[name] = model_var
  init_vars = tf.train.list_variables(ckpt_to_initialize_from)
  assignment_map = collections.OrderedDict()
  var_names = [v.name[:-2] for v in
               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, l2_vars)]
  for variable in init_vars:
    name, _ = variable[0], variable[1]
    if name not in var_names:
      continue
    assignment_map[name] = model_vars_by_name[new_scope + "/" + name + ":0"]
  tf.train.init_from_checkpoint(ckpt_to_initialize_from,
                                assignment_map)
