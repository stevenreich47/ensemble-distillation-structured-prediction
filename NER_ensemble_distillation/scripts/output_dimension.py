""" Computes the output dimension """

# pylint: disable=invalid-name

import argparse as ap
import pickle
import sys

if __name__ == "__main__":
  p = ap.ArgumentParser()
  p.add_argument('--label-map', required=True)
  args = p.parse_args()

  with open(args.label_map, 'rb') as handle:
    labels = pickle.load(handle)

  dimension = str(len(labels.keys()))
  sys.stdout.write(dimension)
