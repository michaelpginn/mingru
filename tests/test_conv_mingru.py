import pytest
import torch
import io

import mingru
from mingru.functional import conv_mingru


def test_conv_mingru_functional():

    rnn = mingru.MinGRU(input_dims=1, hidden_dims=5, num_layers=2)

    x = torch.randn(1, 10, 2, 32, 32)
    h0 = torch.zeros(1, 1, 5, 32, 32)

    h = mingru.functional.g(h0)  # <- don't forget the activation

    kernel = torch.randn(5 * 2, 2, 3, 3)
    bias = torch.randn(5 * 2)

    conv_mingru(x, h, kernel, bias)
