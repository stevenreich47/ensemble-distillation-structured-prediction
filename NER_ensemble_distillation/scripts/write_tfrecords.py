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
Write TFRecords in a given data format. Input file format must
have a corresponding `Parser` defined in `Registries.parsers`.
"""

import argparse
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Type

import tensorflow.compat.v1 as tf

from ner.parser import Parser
from ner.registry import Registries


def _parse_parser_args(extra_vars: Iterable[str]) -> Dict[str, str]:
  vars_list: Dict[str, str] = {}
  for i in extra_vars:
    items: List[str] = i.split('=')
    key = items[0].strip()
    if len(items) > 1:
      value = '='.join(items[1:])
      vars_list[key] = value
  return vars_list


def _parse_args() -> Dict[str, Any]:
  """
  Parse arguments
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-file', required=True)
  parser.add_argument('--data-format', required=True)
  parser.add_argument('--parser', required=True)
  parser.add_argument('--max-sentence-len', type=int, default=None)
  parser.add_argument('--parser-args',
                      nargs='+',
                      metavar="KEY=VALUE",
                      help="Add key/value params. May appear multiple times.")

  args, _ = parser.parse_known_args()
  arg_map: Dict[str, Any] = vars(args)
  parser_args: Dict[str, str] = _parse_parser_args(args.parser_args)
  arg_map['parser_args'] = parser_args
  return arg_map


def write_examples(examples: Iterator[tf.train.SequenceExample],
                   output_file: str):
  n_written = 0
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  with tf.io.TFRecordWriter(output_file) as file:
    for example in examples:
      file.write(example.SerializeToString())
      n_written += 1
  print(f"Wrote {n_written} examples to {output_file}")


def build_parser(parser_name, data_format, parser_args) -> Parser:
  parser_constr: Type[Parser] = Registries.parsers[parser_name]
  return parser_constr(data_format_str=data_format, **parser_args)


def write_tfrecords(output_file: str, data_format: str,
                    parser: str, max_sentence_len: Optional[int],
                    parser_args: Dict[str, Any]) -> None:
  """Writes TF records to the given output_file in the given data_format"""
  in_file = parser_args.pop('conll')
  if max_sentence_len:
    parser_args['max_sentence_len'] = max_sentence_len
  parser_obj = build_parser(parser, data_format, parser_args)
  write_examples(parser_obj(in_file), output_file)


if __name__ == "__main__":
  write_tfrecords(**_parse_args())
