# Copyright 2019 Johns Hopkins University
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
# ============================================================================


"""Methods for dealing with CoNLL IOB2 format NER annotations"""


def get_conll_tag_map():
  """ Return a map of all NER tags """

  return {
      'O': 0,
      'PER': 1,
      'ORG': 2,
      'GPE': 3,
      'LOC': 4,
      'MISC': 5,
      'FAC': 6,
      'TTL': 7
  }


def get_reverse_tag_map():
  """ Inverted tag map """
  rev_map = {}
  fwd_map = get_conll_tag_map()

  for key, value in fwd_map.items():
    rev_map[value] = key
  return rev_map


def get_cheap_conll_maps():
  """ This function is deprecated and will be removed """
  l_map = {}
  r_map = {}

  for key in sorted(list(get_conll_tag_map().keys())):

    if key == 'O':
      r_map[len(l_map)] = key
      l_map[key] = len(l_map)
    else:
      r_map[len(l_map)] = "B-{}".format(key)
      l_map["B-{}".format(key)] = len(l_map)
      r_map[len(l_map)] = "I-{}".format(key)
      l_map["I-{}".format(key)] = len(l_map)

  return l_map, r_map


def is_begin_tag(tag: str) -> bool:
  """ Simple helper """
  return tag.startswith('B-')


def is_inside_tag(tag: str) -> bool:
  """ Simple helper """
  return tag.startswith('I-')


def get_tag_type(tag: str) -> str:
  """ Simple helper """
  return tag.split('-')[-1]
