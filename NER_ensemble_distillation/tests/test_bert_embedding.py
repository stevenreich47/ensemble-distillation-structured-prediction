from typing import Dict

import tensorflow.compat.v1 as tf

from griffin.bert_embedding import BERT
from griffin.features import Features
from griffin.hparam import HParams


class BertEmbeddingTest(tf.test.TestCase):

  @staticmethod
  def test_embed():
    hparams = HParams(
        align=False,
        bert_frozen=True,
        bert_hidden_size=768,
        bert_intermediate_size=3_072,
        bert_num_attention_heads=12,
        bert_num_hidden_layers=12,
        bert_type_vocab_size=2,
        bert_regularize=True,
        bert_vocab_size=119_547,
        birnn_layers=1,
        dropout_keep_prob=0.5,
        grad_clip_norm=1.0,
        hidden_size=256,
        learning_rate=0.05,
        optimizer='adafactor',
        output_dim=9,
        sliding_window_context=1,
        sliding_window_length=3,
        use_bert_layers='-1,-2,-3,-4',
        use_lrate_none=True,
        vars_to_opt=r"^(?!bert).*$",
    )

    bert = BERT(hparams)
    inputs: Dict[str, tf.Tensor] = {
        Features.INPUT_SEQUENCE_LENGTH.value: tf.constant([3, 3],
                                                          dtype=tf.int32),
        Features.INPUT_SYMBOLS.value: tf.constant([[0, 1, 2],
                                                   [3, 4, 5]],
                                                  dtype=tf.int32),
    }
    result: tf.Tensor = bert.embed(inputs=inputs, is_training=False)

    tf.assert_equal(result.shape, (2, 3, 3_072))

    # only check the first and last 3 values (because we're lazy)
    flattened_result = tf.reshape(result, [-1])
    # print(flattened_result)
    tf.assert_near(flattened_result[:3],
                   [1.4477334, -0.19289067, 1.2882688])
    tf.assert_near(flattened_result[-3:],
                   [-0.21256351, -1.8319685, 0.23094822])
