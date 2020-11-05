# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import h5py
import numpy as np
import torch

from fairseq.data import data_utils
from fairseq.data import FairseqDataset
from fairseq.data import LanguagePairDataset

from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_torch_pad(values):
    return pad_sequence(values, batch_first=True)


def collate_3d(values):
    """Input: List of N 3d tensors with different lengths and uniform
    remaining dims D1, D2
    
    Output: 4d tensor with shape [N, T, D1, D2]

    """
    max_len = max(v.size(0) for v in values)
    res = torch.zeros(len(values),
                      max_len,
                      values[0].size(1),
                      values[0].size(2)).type(values[0].type())
    res = res * 5  # debugging XXXXX

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        t = v.size(0)
        copy_tensor(v, res[i][:t])

    return res


def collate_2d(values):
    """Input: List of N 2d tensors with different lengths and uniform
    last dim D
    
    Output: 3d tensor with shape [N, T, D]

    """
  
    max_len = max(v.size(0) for v in values)
    last_dim = values[0].size(1)
    res = torch.zeros(len(values), max_len, last_dim)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        t = v.size(0)
        copy_tensor(v, res[i][:t])

    return res


def collate_generic(values, left_pad=False):
    """Convert a generic list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(0)

    def copy_tensor(src, dst):
        # numel(): Returns the total number of elements in the input tensor.
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])

    return res


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    teacher_inds = None
    teacher_vals = None
    if samples[0].get('teacher_inds', None) is not None:
        # Note: these are flattened; we need to unflatten to [B, L, K]
        ndim = samples[0]['teacher_inds'].dim()
        if ndim == 3:
          inds = []
          vals = []
          for val, ind in zip([s['teacher_vals'] for s in samples],
                              [s['teacher_inds'] for s in samples]):
            inds.append(ind)
            vals.append(val)
          teacher_inds = collate_torch_pad(inds).index_select(0, sort_order)
          teacher_vals = collate_torch_pad(vals).index_select(0, sort_order)
        elif ndim == 2:
          teacher_inds = collate_2d([s['teacher_inds'] for s in samples]).index_select(0, sort_order)
          teacher_vals = collate_2d([s['teacher_vals'] for s in samples]).index_select(0, sort_order)
        elif ndim == 1:
          teacher_inds = collate_generic([s['teacher_inds'] for s in samples]).index_select(0, sort_order)
          teacher_vals = collate_generic([s['teacher_vals'] for s in samples]).index_select(0, sort_order)
        else:
          raise ValueError(ndim)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'teacher_vals': teacher_vals,
        'teacher_inds': teacher_inds
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class LanguagePairDatasetWithTeacher(LanguagePairDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        teacher_file: file with teacher predictions
        top_k: top k values from experts

    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False, eos=None,
        teacher_file=None, top_k=None
    ):
        super().__init__(src, src_sizes, src_dict, tgt=tgt,
                         tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
                         left_pad_source=left_pad_source,
                         left_pad_target=left_pad_target,
                         max_source_positions=max_source_positions,
                         max_target_positions=max_target_positions,
                         shuffle=shuffle, input_feeding=input_feeding,
                         remove_eos_from_source=remove_eos_from_source,
                         append_eos_to_target=append_eos_to_target,
                         align_dataset=align_dataset, append_bos=append_bos,
                         eos=eos)

        print(f"Left pad source? {left_pad_source}")
        print(f"Left pad target? {left_pad_target}")

        self.top_k = top_k
        self.teacher_file = teacher_file
        self.teacher = None

    def open_teacher_file(self):
        if self.teacher_file:
          print(f"Loading teacher file: {self.teacher_file}")
          if not self.teacher_file.endswith('.h5'):
            raise ValueError(f"Expected h5 extension: {self.teacher_file}")

          files = self.teacher_file.split(':')

          if len(files) == 1:
            self.teacher = h5py.File(self.teacher_file, 'r')
            self.teacher_vals = self.teacher["lprobs"]
            self.teacher_inds = self.teacher["indices"]
          else:
            self.teacher = []
            self.teacher_vals = []
            self.teacher_inds = []
            for file_ in files:
              handle = h5py.File(file_, 'r')
              self.teacher_vals.append(handle["lprobs"])
              self.teacher_inds.append(handle["indices"])
              self.teacher.append(handle)  # so we keep it open
        else:
          self.teacher = None
          print("No teacher file")

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `teacher_inds` (LongTensor): a padded 2D Tensor of tokens in
                  the top K of the teacher predictive distribution
                - `teacher_vals` (LongTensor): a padded 2D Tensor of log
                  probabilities in the top K of the teacher predictive distribution
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item
        }

        if self.teacher_file and self.teacher is None:
            self.open_teacher_file()
        
        if self.teacher is not None:
            if isinstance(self.teacher, list):
                # time x expert x event
                inds = [teacher_inds[index] for teacher_inds in self.teacher_inds]
                for ind in inds:
                  assert ind.reshape(-1, 64).shape[0] == tgt_item.shape[0], f"{ind.reshape(-1, 64).shape[0]} {tgt_item.shape[0]}"
                example['teacher_inds'] = torch.stack(
                  [torch.LongTensor(i).reshape(-1, self.top_k) for i in inds], 1)

                vals = [teacher_vals[index] for teacher_vals in self.teacher_vals]
                for val in vals:
                  assert val.reshape(-1, 64).shape[0] == tgt_item.shape[0], f"{val.reshape(-1, 64).shape[0]} {tgt_item.shape[0]}"
                example['teacher_vals'] = torch.stack(
                  [torch.FloatTensor(v).reshape(-1, self.top_k) for v in vals], 1)
            else:
                example['teacher_inds'] = torch.tensor(self.teacher_inds[index]).long()
                example['teacher_vals'] = torch.tensor(self.teacher_vals[index]).float()
        
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example


if __name__ == "__main__":
    filename = "/expscratch/nandrews/nmt/fairseq/jobs/scaling_nmt/e1b/train_e1b_dist.h5"
    file_handle = h5py.File(filename, 'r')
    vals_ds = file_handle["lprobs"]
    inds_ds = file_handle["indices"]
    import numpy as np

    sample = []
    for i in np.random.randint(1000, size=(5)):
      sample.append({
        'vals': torch.tensor(vals_ds[i]).float(),
        'inds': torch.tensor(inds_ds[i]).long()
      })

    result = collate_generic([s['vals'] for s in sample])
    print(result.shape)
    print(result[0][:5])
    result = collate_generic([s['inds'] for s in sample])
    print(result.shape)
    print(result[0][:5])
