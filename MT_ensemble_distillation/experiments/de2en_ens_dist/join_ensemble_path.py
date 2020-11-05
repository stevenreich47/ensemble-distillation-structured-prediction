import sys
import os
import argparse


def ensemble_path(root, models):
  paths = [os.path.join(root, model) for model in models]
  for path in paths:
    if not os.path.exists(path):
      raise ValueError(path)
  return ":".join(paths)


if __name__ == "__main__":
  root = sys.argv[1]
  models = sys.argv[2:]
  parser = argparse.ArgumentParser()
  parser.add_argument('root', type=str,
                      help='')
  parser.add_argument('paths', type=str, nargs='+',
                      help='checkpoints or job directories')
  parser.add_argument('--checkpoint', choices=['last', 'specified', 'avg'],
                      default='specified',
                      help='which checkpoint to use (meaning of paths)')

  args = parser.parse_args()

  if args.checkpoint == 'last':
    models = [os.path.join(path, 'checkpoint_last.pt') for path in args.paths]
  elif args.checkpoint == 'avg':
    models = [os.path.join(path, 'checkpoint_avg.pt') for path in args.paths]
  else:
    models = args.paths
  
  print(ensemble_path(args.root, models))
