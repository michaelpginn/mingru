import pytest
import torch

import mingru


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_mingru(num_layers):

    rnn = mingru.MinGRU(input_dims=3, hidden_dims=5, num_layers=num_layers)

    h0 = torch.zeros(2, 1, 5)
    x = torch.randn(2, 5, 3)

    h_seq = [mingru.g(h0)]  # <- don't forget the activation
    for i in range(5):
        hn = rnn(x[:, i : i + 1], h_seq[-1])
        h_seq.append(hn)
    h_seq = torch.cat(h_seq, 1)

    h_par = rnn(x, mingru.g(h0))
    assert h_par.shape == (2, 5, 5)
    assert torch.allclose(h_seq[:, 1:], h_par)

    h_par = rnn(x, mingru.g(h0), return_all_outputs=True)
    assert h_par.shape == (num_layers, 2, 5, 5)
    assert torch.allclose(h_seq[:, 1:], h_par[-1])


def test_interface():

    # Instantiate
    rnn = mingru.MinGRU(input_dims=3, hidden_dims=64, num_layers=2)

    # Invoke without initial hidden state (will allocate zero hidden state)
    h = rnn(torch.randn(10, 128, 3))
    assert h.shape == (10, 128, 64)

    # Invoke without initial hidden state and get all intermediate hidden states
    h = rnn(torch.randn(10, 128, 3), return_all_outputs=True)
    assert h.shape == (2, 10, 128, 64)

    # Invoke with initial hidden state
    h = rnn(torch.randn(10, 128, 3), h=torch.ones(10, 1, 64) * 0.5)
    assert h.shape == (10, 128, 64)

    # For sequential prediction use
    data = torch.randn(10, 128, 3)
    h0 = torch.ones(10, 1, 64) * 0.5
    res = [h0]
    for i in range(128):
        h = rnn(data[:, i : i + 1], res[-1])
        res.append(h)

    # Note, don't use all-zeros for initial hidden state, instead
    # use the activation function
    h = rnn(torch.randn(10, 128, 3), mingru.g(torch.zeros(10, 1, 64)))
    assert h.shape == (10, 128, 64)
