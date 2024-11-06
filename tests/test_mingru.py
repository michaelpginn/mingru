import torch

from mingru import MinGRU, g


def test_mingru():

    mingru = MinGRU(input_dims=3, hidden_dims=5)

    h0 = torch.zeros(2, 5)
    x = torch.randn(2, 5, 3)

    h_seq = [g(h0)]  # <- don't forget the activation
    for i in range(5):
        hn = mingru(x[:, i], h_seq[-1])
        h_seq.append(hn)
    h_seq = torch.stack(h_seq, 1)

    h_par = mingru(x, g(h0).view(2, 1, 5))

    assert torch.allclose(h_seq[:, 1:], h_par)
