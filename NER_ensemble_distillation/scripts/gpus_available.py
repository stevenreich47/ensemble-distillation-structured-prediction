"""Just prints out the number of GPUs available
(as a check that GPUs will be used)
"""

from __future__ import absolute_import, division, print_function, \
  unicode_literals

import tensorflow as tf

print("Num GPUs Available: ",
      len(tf.config.experimental.list_physical_devices('GPU')))
