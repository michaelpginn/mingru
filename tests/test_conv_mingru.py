import pytest
import torch
import io

import mingru
from mingru.functional import conv_mingru


def test_conv_mingru_functional():

    x = torch.randn(1, 10, 2, 32, 32)
    h0 = torch.zeros(1, 1, 5, 32, 32)

    kernel = torch.randn(5 * 2, 2, 3, 3)
    bias = torch.randn(5 * 2)

    h = mingru.functional.g(h0)  # <- don't forget the activation

    out_seq = []
    for i in range(x.shape[1]):
        # For more than 1 layers we need all intermediate hidden states
        # for next invocation
        h = conv_mingru(x[:, i : i + 1], h, kernel, bias)
        print(h.shape)
        out_seq.append(h)
    out_seq = torch.cat(out_seq, 1)
