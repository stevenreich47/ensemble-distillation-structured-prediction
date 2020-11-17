#!/usr/bin/env python

"""Simple script that copies CoNLL files, adding 4 random floats in [0,1]
to each token line (to represent bounding boxes)"""

import os
import random
from pathlib import Path
from typing import List

import click


@click.command()
@click.argument('conll-files', type=click.Path(), nargs=-1)
@click.option('--out-dir', '-o', default='output',
              type=click.Path(file_okay=False),
              help='directory into which modified files will be written')
def generate_bounding_boxes(out_dir: str, conll_files: List[str]) -> None:
  """Main entry point function for this script"""
  os.makedirs(out_dir, exist_ok=True)
  for f in conll_files:
    in_path = Path(f)
    click.echo(f'Adding bounding boxes to {in_path.name}...')
    out_path = Path(out_dir, in_path.name)
    click.echo(f'\tWrote output to {out_path}')
    with open(in_path) as in_file:
      with open(out_path, mode='w') as out_file:
        _do_one_file(in_file, out_file)


def _rand_box() -> List[str]:
  return [f'{random.uniform(0, 1):.4f}' for _ in range(4)]


def _do_one_file(in_file, out_file) -> None:
  """Process one input file
  @in_file the input file to copy/modify
  @out_file the output file to write the results to"""
  for line in in_file:
    fields: List[str] = line.rstrip().split('\t')
    n_fields = len(fields)
    if n_fields == 0:
      out_line = ''  # sentence break
    elif n_fields == 1:
      out_line = fields[0]  # -DOCSTART- line, e.g.
    else:
      # some data files have extra fields, which for this, we need to remove
      out_line = '\t'.join(fields[:2] + _rand_box())
    out_file.write(out_line + '\n')


if __name__ == '__main__':
  generate_bounding_boxes()  # pylint: disable=no-value-for-parameter
