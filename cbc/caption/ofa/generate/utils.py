# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
#

import collections
from itertools import accumulate

import torch
import torch.nn.functional as F

try:
    from amp_C import multi_tensor_l2norm  # noqa: F401

    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


MANIFOLD_PATH_SEP = "|"


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict((key, _apply(value)) for key, value in x.items())
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def get_token_to_word_mapping(tokens, exclude_list):
    n = len(tokens)
    word_start = [int(token not in exclude_list) for token in tokens]
    word_idx = list(accumulate(word_start))
    token_to_word = {i: word_idx[i] for i in range(n)}
    return token_to_word


def extract_hard_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = ((tgt_sent != pad) & (tgt_sent != eos)).nonzero(as_tuple=False).squeeze(dim=-1)
    src_invalid = ((src_sent == pad) | (src_sent == eos)).nonzero(as_tuple=False).squeeze(dim=-1)
    src_token_to_word = get_token_to_word_mapping(src_sent, [eos, pad])
    tgt_token_to_word = get_token_to_word_mapping(tgt_sent, [eos, pad])
    alignment = []
    if len(tgt_valid) != 0 and len(src_invalid) < len(src_sent):
        attn_valid = attn[tgt_valid]
        attn_valid[:, src_invalid] = float("-inf")
        _, src_indices = attn_valid.max(dim=1)
        for tgt_idx, src_idx in zip(tgt_valid, src_indices):
            alignment.append(
                (
                    src_token_to_word[src_idx.item()] - 1,
                    tgt_token_to_word[tgt_idx.item()] - 1,
                )
            )
    return alignment


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


def extract_soft_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = (tgt_sent != pad).nonzero(as_tuple=False)
    src_valid = (src_sent != pad).nonzero(as_tuple=False).squeeze(dim=-1)
    alignment = []
    if len(tgt_valid) != 0 and len(src_valid) != 0:
        attn_valid = attn[tgt_valid, src_valid]
        alignment = [[f"{p:.6f}" for p in src_probs.tolist()] for src_probs in attn_valid]
    return alignment
