import pytest
import torch

from mingru import MinGRU, g


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_mingru2(num_layers):

    mingru = MinGRU(input_dims=3, hidden_dims=5, num_layers=num_layers)

    h0 = torch.zeros(2, 1, 5)
    x = torch.randn(2, 5, 3)

    h_seq = [g(h0)]  # <- don't forget the activation
    for i in range(5):
        hn = mingru(x[:, i : i + 1], h_seq[-1])
        h_seq.append(hn)
    h_seq = torch.cat(h_seq, 1)

    h_par = mingru(x, g(h0))
    assert h_par.shape == (2, 5, 5)
    assert torch.allclose(h_seq[:, 1:], h_par)

    h_par = mingru(x, g(h0), return_all_outputs=True)
    assert h_par.shape == (num_layers, 2, 5, 5)
    assert torch.allclose(h_seq[:, 1:], h_par[-1])
