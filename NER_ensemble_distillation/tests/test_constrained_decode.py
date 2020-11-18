from itertools import product

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
from tensorflow.python.framework import test_util

from scipy.special import logsumexp

from ner.decode import fwd

from ner.confidences import span_tuples
from ner.crf import forward
from ner.crf import log_norm
from ner.crf import span_confidence
from ner.crf import token_confidence


def satisfies_constraints(states, constraints):
  if not constraints:
    return True

  for constraint in constraints:
    index, state = constraint
    if states[index] != state:
      return False

  return True


def exact_partition(unary, binary, num_states, length, constraints=None):
  zs = []
  for states in product(range(num_states), repeat=length):
    if satisfies_constraints(states, constraints):
      zs += [sum([unary[t][states[t]] + binary[states[t - 1]][states[t]]
                  for t in range(1, length)]) + unary[0][states[0]]]
  return logsumexp(zs)


class DecodeTest(tf.test.TestCase):

  def test_span_tuples(self):
    r1 = span_tuples(['B-X', 'I-X', 'O'])
    assert len(r1) == 2
    r2 = span_tuples(['B-X', 'B-X'])
    assert len(r2) == 2
    r3 = span_tuples(['O', 'B-X', 'I-X', 'B-X', 'I-X'])
    assert len(r3) == 3
    r4 = span_tuples(['B-X', 'B-Y'])
    assert len(r4) == 2
    r5 = span_tuples(['B-X', 'B-Y', 'I-Y'])
    assert len(r5) == 2
    r6 = span_tuples(['B-X', 'O', 'B-X'])
    assert len(r6) == 3

  def test_span_confidence(self):
    labels = ['O', 'B-X', 'I-X', 'B-Y']
    LENGTH = len(labels)
    label_map = {'O': 0,
                 'B-X': 1,
                 'I-X': 2,
                 'B-Y': 3,
                 'I-Y': 4}
    NUM_STATES = len(label_map)

    # numpy inputs
    np.random.seed(42)
    scores = np.random.uniform(size=(LENGTH, NUM_STATES)).astype(np.float32)
    transition_params = np.random.uniform(size=(NUM_STATES, NUM_STATES)).astype(np.float32)

    # check eager execution
    CONSTRAINTS = [(0, 1), (1, 1)]
    exact_result = exact_partition(scores, transition_params,
                                   NUM_STATES, LENGTH, CONSTRAINTS)
    state_mask = np.ones([1, LENGTH, NUM_STATES], dtype=np.float32)
    state_mask[0][0][:] = -np.inf
    state_mask[0][0][1] = 1
    state_mask[0][1][:] = -np.inf
    state_mask[0][1][1] = 1
    print('getting eager log norm...')
    forward_result = tf.squeeze(log_norm(
      tf.expand_dims(tf.constant(scores), 0),
      tf.expand_dims(tf.constant(np.array(LENGTH, dtype=np.int32)), 0),
      tf.constant(transition_params),
      tf.constant(state_mask)))
    self.assertAllClose(forward_result, exact_result)

    scores_2 = np.random.uniform(size=(LENGTH, NUM_STATES)).astype(np.float32)
    state_mask_2 = np.ones([1, LENGTH, NUM_STATES], dtype=np.float32)
    exact_result_2 = exact_partition(scores_2, transition_params,
                                     NUM_STATES, LENGTH - 2)

    batch_forward_result = tf.squeeze(log_norm(
      tf.constant(np.stack([scores, scores_2], 0)),
      tf.constant(np.array([LENGTH, LENGTH - 2], dtype=np.int32)),
      tf.constant(transition_params),
      state_mask=tf.constant(np.concatenate([state_mask, state_mask_2], 0))))

    self.assertAllClose(batch_forward_result[0], exact_result)
    self.assertAllClose(batch_forward_result[1], exact_result_2)

    log_probs = span_confidence(
      labels,
      label_map,
      scores,
      tf.constant(transition_params))

    Z = exact_partition(scores, transition_params,
                        NUM_STATES, LENGTH)

    exact_log_probs = [
      exact_partition(scores, transition_params,
                      NUM_STATES, LENGTH, [(0, 0)]) - Z,
      exact_partition(scores, transition_params,
                      NUM_STATES, LENGTH, [(1, 1), (2, 2)]) - Z,
      exact_partition(scores, transition_params,
                      NUM_STATES, LENGTH, [(1, 1), (2, 2)]) - Z,
      exact_partition(scores, transition_params,
                      NUM_STATES, LENGTH, [(3, 3)]) - Z
    ]

    for i in range(4):
      self.assertAllClose(log_probs[i], exact_log_probs[i])

    # Length 1
    LENGTH = 1

    scores = np.random.uniform(size=(LENGTH, NUM_STATES)).astype(np.float32)
    transition_params = np.random.uniform(size=(NUM_STATES, NUM_STATES)).astype(np.float32)
    Z = exact_partition(scores, transition_params,
                        NUM_STATES, LENGTH)
    exact_log_probs = [
      exact_partition(scores, transition_params,
                      NUM_STATES, LENGTH, [(0, 0)]) - Z
    ]

    log_probs = span_confidence(
      ['O'],
      label_map,
      scores,
      tf.constant(transition_params))

    for i in range(len(exact_log_probs)):
      self.assertAllClose(log_probs[i], exact_log_probs[i])

    # Length 1
    print("Using exact scores")
    
    LENGTH = 1

    label_map = {'O': 0}
    for i in range(1, 9):
      label_map[f"{i}"] = i
    NUM_STATES = len(label_map)

    scores = np.array([[17.210318,  -1.7523448, -1.7468946, -1.7541629, -1.745362,  -1.6750281, -0.6549013, -1.0089291, -1.7052418]]).astype(np.float32)
    transition_params = np.array([[3.8820984, 2.0081537, -3.320762, 3.3802872, -2.4926484, 1.0351412, -4.284433, 2.7741752, -2.921692],
                [0.10092355, -3.015394, 5.400728, 1.2779013, -2.290202,  -2.250833, -4.3584757, -0.6566979, -1.7163708],
                [-1.4402012, -2.744968, 6.998557, 0.44674885, -1.197187, -0.94005746, -2.465937, 0.09801143, -0.57973826],
                [1.5439456, -1.5481476, -1.6593307, -0.54902786, 9.808681, -1.2565507, -1.2481883, -0.638866, -1.9015613],
                [-0.73901147, -1.6916517, -1.2796685, -0.6903837, 9.922528, -0.33926877, -0.69750476, -0.29245436, -0.88925594],
                [0.5474703, -3.688774, -2.6614795, -0.6659309, -0.49762172, -5.379544, 3.6994147, -0.2626439, -0.68241453],
                [-4.3378053, -2.8158345, -2.0620396, -1.5432757, -0.33419633, -4.7769136, 3.0754242, -0.5218125, -0.7228858],
                [1.2908363, -0.29540873, -1.0041519, 1.3427248, -1.0656849, -0.83056146, -1.1367968, -3.1452491, 7.3495727],
                [-1.2124475, -0.57160926, -0.5560673, 0.31276995, -0.83129334, 0.2427108, -0.72370785, -2.386071, 7.0763965]]).astype(np.float32)

    Z = exact_partition(scores, transition_params,
                        NUM_STATES, LENGTH)
    exact_log_probs = [
      exact_partition(scores, transition_params,
                      NUM_STATES, LENGTH, [(0, 0)]) - Z
    ]

    log_probs = span_confidence(
      ['O'],
      label_map,
      scores,
      tf.constant(transition_params))

    self.assertAllClose(log_probs[0], exact_log_probs[0])


  def test_get_token_probs(self):
    T = 5
    K = 3
    np.random.seed(42)
    scores = np.random.uniform(size=(T, K)).astype(np.float32)
    transition_params = np.random.uniform(size=(K, K)).astype(np.float32)
    z = exact_partition(scores, transition_params, K, T)
    exact_log_probs = [exact_partition(scores, transition_params, K, T, [C])
                       for C in product(range(T), range(K))]
    exact_log_probs = np.array(exact_log_probs).reshape((T, K)) - z
    log_probs = token_confidence(scores, transition_params)
    for t in range(T):
      for k in range(K):
        self.assertAllClose(exact_log_probs[t][k], log_probs[t][k])

    # Length 1, random scores + trans params
    T = 1
    K = 9
    scores = np.random.uniform(size=(T, K)).astype(np.float32)
    transition_params = np.random.uniform(size=(K, K)).astype(np.float32)

    z = exact_partition(scores, transition_params, K, T)
    exact_log_probs = [exact_partition(scores, transition_params, K, T, [C])
                       for C in product(range(T), range(K))]
    exact_log_probs = np.array(exact_log_probs).reshape((T, K)) - z
    log_probs = token_confidence(scores, transition_params)
    for t in range(T):
      for k in range(K):
        self.assertAllClose(exact_log_probs[t][k], log_probs[t][k])

    # Length 1, explicit scores + trans params from trained model
    scores = np.array([[17.210318,  -1.7523448, -1.7468946, -1.7541629, -1.745362,  -1.6750281, -0.6549013, -1.0089291, -1.7052418]]).astype(np.float32)
    transition_params = np.array([[3.8820984, 2.0081537, -3.320762, 3.3802872, -2.4926484, 1.0351412, -4.284433, 2.7741752, -2.921692],
                [0.10092355, -3.015394, 5.400728, 1.2779013, -2.290202,  -2.250833, -4.3584757, -0.6566979, -1.7163708],
                [-1.4402012, -2.744968, 6.998557, 0.44674885, -1.197187, -0.94005746, -2.465937, 0.09801143, -0.57973826],
                [1.5439456, -1.5481476, -1.6593307, -0.54902786, 9.808681, -1.2565507, -1.2481883, -0.638866, -1.9015613],
                [-0.73901147, -1.6916517, -1.2796685, -0.6903837, 9.922528, -0.33926877, -0.69750476, -0.29245436, -0.88925594],
                [0.5474703, -3.688774, -2.6614795, -0.6659309, -0.49762172, -5.379544, 3.6994147, -0.2626439, -0.68241453],
                [-4.3378053, -2.8158345, -2.0620396, -1.5432757, -0.33419633, -4.7769136, 3.0754242, -0.5218125, -0.7228858],
                [1.2908363, -0.29540873, -1.0041519, 1.3427248, -1.0656849, -0.83056146, -1.1367968, -3.1452491, 7.3495727],
                [-1.2124475, -0.57160926, -0.5560673, 0.31276995, -0.83129334, 0.2427108, -0.72370785, -2.386071, 7.0763965]]).astype(np.float32)

    z = exact_partition(scores, transition_params, K, T)
    exact_log_probs = [exact_partition(scores, transition_params, K, T, [C])
                       for C in product(range(T), range(K))]
    exact_log_probs = np.array(exact_log_probs).reshape((T, K)) - z
    log_probs = token_confidence(scores, transition_params)
    for t in range(T):
      for k in range(K):
        self.assertAllClose(exact_log_probs[t][k], log_probs[t][k])


  @test_util.deprecated_graph_mode_only
  def test_length_one(self):
    NUM_STATES = 3
    LENGTH = 1
    CONSTRAINTS = [(0, 1)]
    np.random.seed(42)
    score = np.random.uniform(size=[LENGTH, NUM_STATES]).astype(np.float32)
    transition_params = np.zeros([NUM_STATES, NUM_STATES], dtype=np.float32)
    exact_0 = exact_partition(score, transition_params,
                              NUM_STATES, LENGTH)
    print('result 0 (exhaustive): ', exact_0)
    exact_1 = exact_partition(score, transition_params,
                              NUM_STATES, LENGTH, CONSTRAINTS)
    print('result 1 (exhaustive, constrained): ', exact_1)

    expanded_score = tf.constant(np.expand_dims(score, 0))  # batch dim
    lens = tf.constant(np.array([LENGTH], dtype=np.int32))
    state_mask = np.ones([1, 1, NUM_STATES], dtype=np.float32)
    state_mask[0][0][0] = -np.inf  # rule out state 0 at pos 0
    state_mask[0][0][2] = -np.inf  # rule out state 2 at pos 0
    forward_0 = tf.squeeze(log_norm(
      expanded_score,
      lens,
      tf.constant(transition_params)))

    forward_1 = tf.squeeze(log_norm(
      expanded_score,
      lens,
      tf.constant(transition_params),
      tf.constant(state_mask)))

    with tf.Session() as sess:
      forward_0_val = sess.run(forward_0)
      print('result 0 (forward): ', forward_0_val)
      self.assertAllClose(exact_0, forward_0_val,
                          msg="Unconstrained length = 1 incorrect")

      forward_1_val = sess.run(forward_1)
      print('result 1 (forward, constrained): ', forward_1_val)
      self.assertAllClose(exact_1, forward_1_val,
                          msg="Constrained length = 1 incorrect")

  @test_util.also_run_as_tf_function
  def test_fwd_v2(self):
    NUM_STATES = 2
    LENGTH = 3
    score = tf.expand_dims(tf.random.uniform(shape=[LENGTH, NUM_STATES]), 0)
    transition_params = tf.zeros([NUM_STATES, NUM_STATES], dtype=tf.float32)
    lens = tf.constant(np.array([LENGTH], dtype=np.int32))
    tf.squeeze(log_norm(
        score,
        lens,
        transition_params))

  @test_util.deprecated_graph_mode_only
  def test_fwd(self):
    NUM_STATES = 2
    LENGTH = 3

    np.random.seed(42)

    score = np.random.uniform(size=[LENGTH, NUM_STATES]).astype(np.float32)
    # The binary scores are interpreted as log probabilities,
    # so that log(0) = 1.0 corresponds to uniform transition
    # probabilities.
    transition_params = np.zeros([NUM_STATES, NUM_STATES], dtype=np.float32)

    exact_val = exact_partition(score, transition_params,
                                NUM_STATES, LENGTH)
    print('result 0 (exhaustive): ', exact_val)
    print(score)
    print(score.shape)
    result = fwd(score)
    print('result 1 (fwd-A): ', result)

    result2 = fwd(score, transition_params=transition_params)
    print('result 2 (fwd-B): ', result2)
    #tf.assert_near(result, result2)

    expanded_score = tf.constant(np.expand_dims(score, 0))
    lens = tf.constant(np.array([LENGTH], dtype=np.int32))
    print(expanded_score.shape)
    result3 = tf.squeeze(tfa.text.crf.crf_log_norm(
      expanded_score,
      lens,
      tf.constant(transition_params)
    ))

    first_input = tf.slice(expanded_score, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])
    rest_of_input = tf.slice(expanded_score, [0, 1, 0], [-1, -1, -1])

    alphas = forward(
      rest_of_input,
      first_input,
      tf.constant(transition_params),
      lens
    )

    assert alphas.shape[0] == 1
    assert alphas.shape[1] == NUM_STATES

    result4 = tf.squeeze(log_norm(
      expanded_score,
      lens,
      tf.constant(transition_params)))

    CONSTRAINTS = [(1, 0)]
    result5_exact = exact_partition(score, transition_params,
                                    NUM_STATES, LENGTH, CONSTRAINTS)
    state_mask = np.ones([1, LENGTH, NUM_STATES], dtype=np.float32)
    state_mask[0][1][1] = -np.inf  # rule out state 0 at pos 0
    result5_forward = tf.squeeze(log_norm(
      expanded_score,
      lens,
      tf.constant(transition_params),
      state_mask))

    CONSTRAINTS = [(0, 1)]
    result6_exact = exact_partition(score, transition_params,
                                    NUM_STATES, LENGTH, CONSTRAINTS)
    state_mask = np.ones([1, LENGTH, NUM_STATES], dtype=np.float32)
    state_mask[0][0][0] = -np.inf  # rule out state 0 at pos 0
    result6_forward = tf.squeeze(log_norm(
      expanded_score,
      lens,
      tf.constant(transition_params),
      state_mask))

    CONSTRAINTS = [(0, 1), (1, 1)]
    result7_exact = exact_partition(score, transition_params,
                                    NUM_STATES, LENGTH, CONSTRAINTS)
    state_mask = np.ones([1, LENGTH, NUM_STATES], dtype=np.float32)
    state_mask[0][0][0] = -np.inf  # rule out state 0 at pos 0
    state_mask[0][1][0] = -np.inf  # rule out state 0 at pos 1
    result7_forward = tf.squeeze(log_norm(
      expanded_score,
      lens,
      tf.constant(transition_params),
      state_mask))

    CONSTRAINTS = [(0, 1), (1, 1), (2, 0)]
    result8_exact = exact_partition(score, transition_params,
                                    NUM_STATES, LENGTH, CONSTRAINTS)
    state_mask = np.ones([1, LENGTH, NUM_STATES], dtype=np.float32)
    state_mask[0][0][0] = -np.inf  # rule out state 0 at pos 0
    state_mask[0][1][0] = -np.inf  # rule out state 0 at pos 1
    state_mask[0][2][1] = -np.inf  # rule out state 1 at pos 2
    result8_forward = tf.squeeze(log_norm(
      expanded_score,
      lens,
      tf.constant(transition_params),
      state_mask))

    # Batch support
    result9_forward = tf.squeeze(log_norm(
      tf.concat([expanded_score, expanded_score], 0),
      tf.concat([lens, lens], 0),
      tf.constant(transition_params),
      np.concatenate([state_mask, state_mask], 0)
    ))

    print('trying crf_log_norm in graph mode')
    with tf.Session() as sess:
      alphas_value = sess.run(alphas)
      print('alphas value:')
      print(alphas_value)
      print(alphas_value.shape)

      result3_value = sess.run(result3)
      print('result 3 (TFA): ', result3_value)

      result4_value = sess.run(result4)
      print('result 4 (crf): ', result4_value)

      result5_forward_val = sess.run(result5_forward)
      result6_forward_val = sess.run(result6_forward)
      result7_forward_val = sess.run(result7_forward)
      result8_forward_val = sess.run(result8_forward)

      print("testing batch")
      result9_forward_val = sess.run(result9_forward)

      self.assertAllClose(result3_value, result4_value)
      self.assertAllClose(result3_value, exact_val)
      self.assertAllClose(result5_exact, result5_forward_val,
                          msg="Constrained mismatch")
      self.assertAllClose(result6_exact, result6_forward_val)
      self.assertAllClose(result7_exact, result7_forward_val)
      self.assertAllClose(result8_exact, result8_forward_val)
      self.assertAllClose(result9_forward_val[0], result8_exact)
      self.assertAllClose(result9_forward_val[1], result8_exact)


if __name__ == "__main__":
  tf.test.main()
