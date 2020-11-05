# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch
from torch import Tensor

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.distillation_cross_entropy import smooth
from fairseq.criterions.distillation_cross_entropy import smooth_partial
from fairseq.criterions.distillation_cross_entropy import partial_kl_div

@register_criterion("nat_loss_with_teacher")
class LabelSmoothedDualImitationCriterionWithTeacher(FairseqCriterion):

    def __init__(self, task, label_smoothing, distill_temperature, teacher_weight, teacher_top_k):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.temperature = distill_temperature
        self.teacher_weight = teacher_weight
        self.K = teacher_top_k
        print(f"Teacher top K: {self.K}")
        print(f"Distill temp: {self.temperature}")
        print(f"Teacher weight: {self.teacher_weight}")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            '--label-smoothing',
            default=0.,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing',
        )
        parser.add_argument('--distill-temperature', default=1.0, type=float,
                            help='Temperature for distillation')
        parser.add_argument('--teacher-weight', default=0.1, type=float,
                            help='Relative weight of the teacher relative to the NLL')
        parser.add_argument('--teacher-top-k', default=32, type=int,
                            help='Top K probabilities in the teacher predictive dist')

    def _compute_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0,
            teacher_vals=None, teacher_inds=None
    ):
        """
            outputs: batch x len x d_model
            targets: batch x len
            masks:   batch x len

            policy_logprob: if there is some policy
                depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        if masks is not None:
            print("pre mask shapes")
            print(outputs.shape)
            print(targets.shape)
            outputs, targets = outputs[masks], targets[masks]
            print("post mask shapes")
            print(outputs.shape)
            print(targets.shape)

        if masks is not None and not masks.any():  # if everything is masked
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (
                    1 - label_smoothing) - mean_ds(logits) * label_smoothing
            else:
                loss = nll_loss

            # Maybe include distillation loss
            if teacher_vals is not None:
                # Calculate the distillation loss
                B = masks.shape[0]
                T = masks.shape[1]
                
                # Logits is already flat
                soft_lprobs = smooth(logits, self.temperature)
                vals = teacher_vals.reshape(B, T, self.K)[masks]
                inds = teacher_inds.reshape(B, T, self.K)[masks]
                soft_targets = smooth_partial(vals, self.temperature)

                # Compute KL[teacher(T), student(T)]
                distill_loss = partial_kl_div(soft_targets.detach(),
                                              soft_lprobs,
                                              inds)

                # According to [1]: "Since the magnitudes of the gradients
                # produced by the soft targets scale as 1/(T^2) it is
                # important to multiply them by T^2 when using both hard and
                # soft targets. This ensures that the relative contributions
                # of the hard and soft targets remain roughly unchanged if the
                # temperature used for distillation is changed while
                # experimenting with meta-parameters."
                distill_loss_scaled = distill_loss * self.temperature ** 2

                # According to [1]: "We found that the best results were
                # generally obtained by using a condiderably lower weight on
                # the [NLL loss]."
                loss = (1. - self.teacher_weight) * loss + self.teacher_weight * distill_loss_scaled

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)

        # Mask-Predict returns:
        #
        # "word_ins": {
        #     "out": word_ins_out, "tgt": tgt_tokens,
        #     "mask": word_ins_mask, "ls": self.args.label_smoothing,
        #     "nll_loss": True
        # },
        # "length": {
        #     "out": length_out, "tgt": length_tgt,
        #     "factor": self.decoder.length_loss_factor
        # }

        # In `nonautoregressive_transformer.py`, from which the Mask-Predict inherits,
        # the `forward_decoder` methods sets `prev_output_tokens` as `output_tokens`.

        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                if obj == 'word_ins':
                    teacher_vals = sample['teacher_vals']
                    teacher_inds = sample['teacher_inds']
                else:
                    teacher_vals = None
                    teacher_inds = None

                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0),
                    teacher_vals=teacher_vals,
                    teacher_inds=teacher_inds
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
            else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
