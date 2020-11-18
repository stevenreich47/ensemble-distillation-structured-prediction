""" Additional/Custom activation functions """

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Activation, LeakyReLU
from tensorflow.compat.v1.keras.utils.generic_utils import get_custom_objects


# Add the GELU function to Keras
def gelu(x):
  """Gaussian Error Linear Unit activation function used in BERT"""
  return 0.5 * x * ( \
        1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


get_custom_objects().update({'gelu': Activation(gelu)})

# Add leaky-relu so we can use it as a string
get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(alpha=0.2))})
